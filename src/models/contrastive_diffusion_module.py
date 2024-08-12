from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from generative.networks.schedulers import Scheduler, DDIMScheduler
from generative.networks.nets.shift_unet import ShiftUNetModel, return_wrap
from generative.utils.misc import unsqueeze_right, unsqueeze_left
import time
from nilearn.image import load_img, new_img_like
import gc
from hydra.utils import to_absolute_path
import os
from tqdm import tqdm
import itertools
import numpy as np
import pickle
from itertools import chain


class ContrastiveDiffusionLitModule(LightningModule):

    def __init__(
            self,
            share_encoder: torch.nn.Module,
            salient_encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            discriminator: torch.nn.Module,
            diffusion_model: torch.nn.Module,
            diffusion_scheduler: Scheduler,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            paths=None,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                'share_encoder',
                'salient_encoder',
                # 'decoder',
                'discriminator',
                'diffusion_model',
                'diffusion_scheduler'
            ]
        )

        self.share_encoder = share_encoder
        self.salient_encoder = salient_encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.diffusion_model = diffusion_model
        self.scheduler = diffusion_scheduler

        # loss function
        # self.criterion = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        self.baseline_represent = list()
        self.baseline_name = list()

        self.count = 0

    def get_loss(self, noise, predicted_noise, weight=None, loss_type="l2", reduction='mean'):
        if loss_type == 'l1':
            loss = (noise - predicted_noise).abs()
        elif loss_type == 'l2':
            if weight is not None:
                loss = weight * (noise - predicted_noise) ** 2
            else:
                loss = (noise - predicted_noise) ** 2
        else:
            raise NotImplementedError

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diffusion_model(x)

    def on_train_start(self) -> None:

        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Tuple[Dict, Dict]) -> [torch.Tensor, Dict]:

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        # Get data
        pt_dict, nc_dict = batch  # mci,ad;cn
        # images = batch['image']
        # labels = batch['label']

        # nc_images = images[labels == 0]
        # pt_images = images[labels != 0]

        # nc step
        nc_images = nc_dict['image']
        nc_share_z, nc_share_mu, nc_share_logvar = self.share_encoder(nc_images)
        nc_salient_z = torch.zeros(nc_images.shape[0], self.salient_encoder.latent_channels).to(nc_images.device)
        nc_represent = torch.cat([nc_share_z, nc_salient_z], dim=1)

        nc_recon = self.decoder(nc_represent)
        nc_recon_mse_loss = self.get_loss(nc_recon, nc_images, reduction='sum')

        nc_share_kl_loss = 0.5 * (
                nc_share_mu.pow(2) + nc_share_logvar.exp() - nc_share_logvar - 1
        )

        # pt step
        pt_images = pt_dict['image']
        pt_share_z, pt_share_mu, pt_share_logvar = self.share_encoder(pt_images)
        pt_salient_z, pt_salient_mu, pt_salient_logvar = self.salient_encoder(pt_images)
        pt_represent = torch.cat([pt_share_z, pt_salient_z], dim=1)

        pt_recon = self.decoder(pt_represent)
        pt_recon_mse_loss = self.get_loss(pt_recon, pt_images, reduction='sum')

        pt_share_kl_loss = 0.5 * (
                pt_share_mu.pow(2) + pt_share_logvar.exp() - pt_share_logvar - 1
        )

        pt_salient_kl_loss = 0.5 * (
                pt_salient_mu.pow(2) + pt_salient_logvar.exp() - pt_salient_logvar - 1
        )

        pt_noise = torch.randn_like(pt_images, device=pt_images.device)
        pt_timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (pt_images.shape[0],),
            device=pt_images.device,
            dtype=torch.long
        )

        pt_images_t = self.scheduler.add_noise(pt_images, pt_noise, pt_timesteps)
        pt_model_output = self.diffusion_model(
            x=pt_images_t, timesteps=pt_timesteps,
            represent=pt_salient_z,
        )
        self.scheduler.shift_coef = self.scheduler.shift_coef.to(pt_images.device)
        pt_shift_coef = unsqueeze_right(self.scheduler.shift_coef[pt_timesteps], pt_images.ndim)
        pt_noise_pred = return_wrap(pt_model_output, pt_shift_coef)
        self.scheduler.weight = self.scheduler.weight.to(pt_images.device)
        pt_weight = unsqueeze_right(self.scheduler.weight[pt_timesteps], pt_images.ndim)
        pt_mse_loss = self.get_loss(pt_noise, pt_noise_pred, pt_weight, reduction='none').sum(dim=[1, 2, 3, 4])

        # discriminator
        split_size = int(len(pt_share_z) / 2)
        pt_share_z_part1, pt_share_z_part2 = pt_share_z[:split_size], pt_share_z[split_size:]
        pt_salient_z_part1, pt_salient_z_part2 = pt_salient_z[:split_size], pt_salient_z[split_size:]

        # In case we have an odd number of target samples
        size = min(len(pt_share_z_part1), len(pt_share_z_part2))
        z1, z2, = pt_share_z_part1[:size], pt_share_z_part2[:size]
        s1, s2 = pt_salient_z_part1[:size], pt_salient_z_part2[:size]

        q_bar = torch.cat(
            [
                torch.cat([s1, z2], dim=1),
                torch.cat([s2, z1], dim=1)
            ]
        )

        q = torch.cat(
            [
                torch.cat([s1, z1], dim=1),
                torch.cat([s2, z2], dim=1)
            ]
        )

        q_bar_score = self.discriminator(q_bar)
        q_score = self.discriminator(q)

        eps = 1e-6
        q_score = q_score.clone().where(q_score == 0, torch.tensor(eps).to(self.device))
        q_score = q_score.clone().where(q_score == 1, torch.tensor(1 - eps).to(self.device))

        q_bar_score = q_bar_score.clone().where(q_bar_score == 0, torch.tensor(eps).to(self.device))
        q_bar_score = q_bar_score.clone().where(q_bar_score == 1, torch.tensor(1 - eps).to(self.device))

        tc_loss = torch.log(q_score / (1 - q_score)).sum(dim=-1)
        discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score)).sum(dim=-1)

        loss = 0.1 * pt_mse_loss.mean()
        loss += nc_recon_mse_loss.mean() + pt_recon_mse_loss.mean()
        loss += (nc_share_kl_loss.mean() + pt_share_kl_loss.mean() + pt_salient_kl_loss.mean())
        loss += tc_loss.mean() + discriminator_loss.mean()

        loss_dict = {
            'nc_recon_mse': nc_recon_mse_loss.mean(),
            'pt_recon_mse': pt_recon_mse_loss.mean(),
            # 'nc_noise_mse': nc_mse_loss.mean(),
            'pt_noise_mse': pt_mse_loss.mean(),
            'nc_share_kl': nc_share_kl_loss.mean(),
            'pt_share_kl': pt_share_kl_loss.mean(),
            'pt_salient_kl': pt_salient_kl_loss.mean(),
            'tc': tc_loss.mean(),
            'dis': discriminator_loss.mean(),
        }

        return loss, loss_dict

    def training_step(
            self, batch: Tuple[Dict, Dict], batch_idx: int
    ) -> torch.Tensor:
        loss, loss_dict = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""

        torch.cuda.empty_cache()
        gc.collect()

    def validation_step(self, batch: Tuple[Dict, Dict], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, loss_dict = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""

        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        torch.cuda.empty_cache()
        gc.collect()

    def test_step(self, batch: Dict, batch_idx: int) -> None:

        image_save_path = os.path.join(to_absolute_path(self.hparams.paths.generate_path), 'nc_pt_sample')
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

        # extra feature
        # pt_images = batch['image']
        # pt_names = batch['file_name']
        # pt_share_z, pt_share_mu, pt_share_sigma = self.share_encoder(pt_images)
        # pt_salient_z, pt_salient_mu, pt_salient_sigma = self.salient_encoder(pt_images)
        #
        # self.baseline_represent.append(torch.cat([pt_share_z, pt_salient_z], dim=1).detach())
        # self.baseline_name.append(pt_names)

        num_inference_steps = 500
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            beta_start=0.0005,
            beta_end=0.0195,
            clip_sample=False,
        )
        ddim_scheduler.set_timesteps(num_inference_steps)

        # Get data
        nc_dict, pt_dict = batch
        x_start = nc_dict['image']
        size = x_start.shape[0]
        device = x_start.device

        # data = np.zeros((size, len(min_values)))
        # for i in range(len(min_values)):
        #     data[:, i] = np.random.uniform(min_values[i], max_values[i], size)

        nc_share_z, nc_share_mu, nc_share_logvar = self.share_encoder(nc_dict['image'])
        nc_salient_z = torch.zeros(
            nc_dict['image'].shape[0], self.salient_encoder.latent_channels
        ).to(nc_dict['image'].device)
        pt_salient_z, pt_salient_mu, pt_salient_logvar = self.salient_encoder(pt_dict['image'])

        # Encoding
        latent_img = x_start.clone()
        for i in tqdm(
                reversed(range(1, ddim_scheduler.num_inference_steps)),
                desc="ddim inverse"
        ):
            # go through the noising process
            t = ddim_scheduler.timesteps[i]
            model_output = self.diffusion_model(
                latent_img,
                timesteps=torch.Tensor((t,)).to(device),
                represent=nc_salient_z,
            )

            latent_img, _ = ddim_scheduler.reversed_step(model_output.pred, t, latent_img)

        # Generate and save file
        recon_img = latent_img.clone()
        for i in tqdm(
                range(ddim_scheduler.num_inference_steps - 1),
                desc="ddim sampling"
        ):
            t = ddim_scheduler.timesteps[i]

            model_output = self.diffusion_model(
                recon_img, timesteps=torch.Tensor((t,)).to(device),
                represent=pt_salient_z,
            )

            self.scheduler.ddim_coef = self.scheduler.ddim_coef.to(recon_img.device)
            ddim_coef = unsqueeze_right(self.scheduler.ddim_coef[t], recon_img.ndim)
            noise_pred = return_wrap(model_output, ddim_coef)
            recon_img, _ = ddim_scheduler.step(noise_pred, t, recon_img)

        # random sample
        # ddpm_img = torch.randn_like(x_start, device=device)
        # print("t:", self.scheduler.timesteps)
        # for t in tqdm(self.scheduler.timesteps, desc="ddpm sampling"):
        #     model_output = self.net(ddpm_img, timesteps=torch.Tensor((t,)).to(device))
        #     ddpm_img, _ = self.scheduler.step(model_output, t, ddpm_img)

        source_data = load_img(self.hparams.paths.reference_file)
        x_start_nc = x_start.cpu().numpy()
        for i in range(size):
            temp_data = new_img_like(source_data, x_start_nc[i].squeeze())
            temp_data.to_filename(os.path.join(image_save_path, f"source_{self.count + i}.nii"))

        latent_img = latent_img.cpu().numpy()
        for i in range(size):
            temp_data = new_img_like(source_data, latent_img[i].squeeze())
            temp_data.to_filename(os.path.join(image_save_path, f"latent_{i}.nii"))

        recon_img = recon_img.cpu().numpy()
        for i in range(size):
            temp_data = new_img_like(source_data, recon_img[i].squeeze())
            temp_data.to_filename(os.path.join(image_save_path, f"recon_ddim_{self.count + i}.nii"))

        self.count += recon_img.shape[0]

    def on_test_epoch_end(self) -> None:

        # self.trainer.strategy.barrier()  # to let other cards to wait
        torch.cuda.empty_cache()
        gc.collect()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.share_encoder = torch.compile(self.share_encoder)
            self.salient_encoder = torch.compile(self.salient_encoder)
            self.discriminator = torch.compile(self.discriminator)
            self.diffusion_model = torch.compile(self.diffusion_model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = self.hparams.optimizer(
            params=itertools.chain(
                self.share_encoder.parameters(),
                self.salient_encoder.parameters(),
                self.discriminator.parameters(),
                self.diffusion_model.parameters(),
            )
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
