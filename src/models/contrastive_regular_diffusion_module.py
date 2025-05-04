from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from generative.networks.schedulers import Scheduler, DDIMScheduler
from generative.networks.nets.shift_unet import ShiftUNetModel, return_wrap
import time
from nilearn.image import load_img, new_img_like
import gc
from hydra.utils import to_absolute_path
import os
from tqdm import tqdm


class ContrastiveDiffusionRegularLitModule(LightningModule):

    def __init__(
            self,
            diffusion_model: torch.nn.Module,
            diffusion_scheduler: Scheduler,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            paths: str = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['diffusion_model', 'diffusion_scheduler'])

        self.diffusion_model = diffusion_model
        self.scheduler = diffusion_scheduler

        # loss function
        self.criterion = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diffusion_model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch) -> torch.Tensor:

        # Get data
        images = batch['image']

        # Generate random noise
        noise = torch.randn_like(images, device=images.device)

        # Create timesteps
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
            dtype=torch.long
        )

        x_t = self.scheduler.add_noise(images, noise, timesteps)
        model_output = self.diffusion_model(x_t, timesteps)
        noise_pred = return_wrap(model_output)

        loss = self.criterion(noise_pred, noise)

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:

        torch.cuda.empty_cache()
        gc.collect()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""

        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        torch.cuda.empty_cache()
        gc.collect()

    def test_step(self, batch: Dict, batch_idx: int) -> None:

        # # 计算loss
        # loss = self.model_step(batch)
        # # update and log metrics
        # self.test_loss(loss)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        image_save_path = to_absolute_path(self.hparams.paths.generate_path)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

        x_start = batch['image']
        size = x_start.shape[0]
        device = x_start.device

        noise = torch.randn_like(x_start, device=device)
        recon_img = noise.clone()
        for t in tqdm(self.scheduler.timesteps, desc="ddpm sampling"):
            model_output = self.diffusion_model(recon_img, timesteps=torch.Tensor((t,)).to(device))
            noise_pred = return_wrap(model_output)
            recon_img, _ = self.scheduler.step(noise_pred, t, recon_img)

        source_data = load_img(self.hparams.paths.reference_file)
        # for i in range(size):
        #     temp_data = new_img_like(source_data, noise[i][0].cpu().numpy())
        #     temp_data.to_filename(os.path.join(image_save_path, f"noise_{i}.nii"))

        for i in range(size):
            temp_data = new_img_like(source_data, recon_img[i][0].cpu().numpy())
            temp_data.to_filename(os.path.join(image_save_path, f"recon_ddpm_{i}.nii"))

    def on_test_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def setup(self, stage: str) -> None:

        if self.hparams.compile and stage == "fit":
            self.diffusion_model = torch.compile(self.diffusion_model)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.hparams.optimizer(params=self.diffusion_model.parameters())
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
