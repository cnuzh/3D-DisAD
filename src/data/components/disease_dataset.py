from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
import json
import os
from src.data.utiles import apply_covariance_correction, uniform, clamp
import joblib
from tqdm import tqdm
import numpy as np
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker


class SingleDataset(Dataset):
    def __init__(
            self,
            root_path,
            pathology_types=None,
            baseline=True,
            info_keys=None,
            remove_covar: bool = True,
            preprocess: [None, str] = 'uniform',
            transform=None,
            cache: bool = True,
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.data_types = pathology_types
        self.baseline = baseline
        self.labels = {'CN': 0, 'MCI': 1, 'AD': 2}
        self.data_list = list()

        if isinstance(pathology_types, str):
            pathology_types = [pathology_types]

        for pathology_type in pathology_types:
            with open(os.path.join(self.root_path, f"subject_{pathology_type}.json"), "r") as f:
                json_data = json.load(f)
                json_data = json_data['train'] if self.baseline else json_data['test']
                self.data_list.extend(
                    [
                        d | {'pathology_type': pathology_type, 'label': self.labels[pathology_type]}
                        for d in json_data
                    ]
                )
                f.close()

        self.info_keys = info_keys
        self.info_keys.extend(["pathology_type", "label"])
        self.remove_covar = remove_covar
        self.preprocess = preprocess
        self.transform = transform

        variable_dict = joblib.load(os.path.join(self.root_path, 'process_variables.joblib'))
        self.masker = variable_dict["transformer"]

        if self.remove_covar:
            self.correct_variables = variable_dict['correct_variables']
            self.normalize_variables = variable_dict["normalize_variables"]

        self.cache = cache
        if self.cache:
            self.data_cache = list()

            for index in tqdm(range(len(self.data_list)), desc="Caching data"):
                sample_covariance = np.array(
                    [
                        [
                            self.data_list[index]['age'],
                            self.data_list[index]['gender']
                        ],
                    ],
                    dtype=np.float32)
                self.data_list[index]['covariance'] = sample_covariance

                source_image = load_img(
                    os.path.join(
                        self.root_path,
                        self.data_list[index]['pathology_type'],
                        self.data_list[index]['file_name']
                    )
                )

                source_data = self.masker.transform_single_imgs(source_image)
                if self.remove_covar:
                    source_data = apply_covariance_correction(source_data, sample_covariance, self.correct_variables)

                if self.preprocess == 'uniform':
                    source_data = uniform(source_data)  # [0, 1]
                elif self.preprocess == 'clamp':
                    source_data = clamp(source_data, -1, 1)  # [-1, 1]

                target_image = self.masker.inverse_transform(source_data)
                target_image = target_image.get_fdata()
                target_image = np.moveaxis(target_image, -1, 0)
                self.data_cache.append(target_image)

    def __getitem__(self, index):

        if self.info_keys is not None:
            sample_info = {k: self.data_list[index][k] for k in self.info_keys}
        else:
            sample_info = dict()

        if self.cache:
            sample_info.update({'image': self.data_cache[index], 'covariance': self.data_list[index]['covariance']})
        else:
            sample_covariance = np.array(
                [
                    [
                        self.data_list[index]['age'],
                        self.data_list[index]['gender']
                    ],
                ],
                dtype=np.float32)

            sample_info.update({'covariance': sample_covariance})

            source_image = load_img(
                os.path.join(
                    self.root_path,
                    self.data_list[index]['pathology_type'],
                    self.data_list[index]['file_name']
                )
            )

            source_data = self.masker.transform_single_imgs(source_image)
            if self.remove_covar:
                source_data = apply_covariance_correction(source_data, sample_covariance, self.correct_variables)

            if self.preprocess == 'uniform':
                source_data = uniform(source_data)  # [0, 1]
            elif self.preprocess == 'clamp':
                source_data = clamp(source_data, -1, 1)  # [-1, 1]

            target_image = self.masker.inverse_transform(source_data)
            target_image = target_image.get_fdata()
            target_image = np.moveaxis(target_image, -1, 0)
            sample_info.update({'image': target_image})

        if self.transform is not None:
            sample_info = self.transform(sample_info)

        return sample_info

    def __len__(self):
        return len(self.data_list)


class CombinedDataset(Dataset):
    def __init__(self, dataset_finite, dataset_infinite):
        self.dataset_finite = dataset_finite
        self.dataset_infinite = dataset_infinite

        self.infinite_index = 0

    def __getitem__(self, index):
        finite_data = self.dataset_finite[index]
        infinite_data = self.dataset_infinite[self.infinite_index]

        self.infinite_index += 1
        if self.infinite_index == len(self.dataset_infinite):
            self.infinite_index = 0

        return finite_data, infinite_data

    def __len__(self):
        return len(self.dataset_finite)
