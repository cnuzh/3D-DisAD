import numpy as np
from sklearn.linear_model import LinearRegression


def data_mask(cn_data):
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1
    among CN participants for each ROI
    :param cn_data: array, control data
    :return: normalized control data & normalized patient data
    """

    mask = np.logical_not(np.all(cn_data == 0, axis=0)).astype(np.float32)

    mask_variables = {"mask": mask}
    return mask_variables


def covariance_correction(cn_data, cn_cov):
    """
    Eliminate the confounding of variate, such as age and sex, from the disease-based changes.
    :param cn_data: array, control data
    :param cn_cov: array, control covariances
    :return: corrected control data & corrected patient data
    """
    min_cov = np.amin(cn_cov, axis=0)
    max_cov = np.amax(cn_cov, axis=0)
    cn_cov = (cn_cov - min_cov) / (max_cov - min_cov)
    beta = np.transpose(LinearRegression().fit(cn_cov, cn_data).coef_)

    correction_variables = {"max_cov": max_cov, "min_cov": min_cov, "beta": beta}
    return correction_variables


def data_normalization(cn_data):
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1
    among CN participants for each ROI
    :param cn_data: array, control data
    :return: normalized control data & normalized patient data
    """

    cn_mean = np.mean(cn_data, axis=0)
    cn_std = np.std(cn_data, axis=0)

    # fix error
    cn_std = np.where(cn_std == 0.0, np.ones_like(cn_std), cn_std)

    normalization_variables = {"cn_mean": cn_mean, "cn_std": cn_std}
    return normalization_variables


def data_min_max_normalization(cn_data):
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1
    among CN participants for each ROI
    :param cn_data: array, control data
    :return: normalized control data & normalized patient data
    """

    data_max = np.max(cn_data, axis=0)
    data_min = np.min(cn_data, axis=0)

    correct_data_max = np.where(
        data_max == data_min, np.ones_like(data_max, dtype=np.float32), data_max
    )
    correct_data_min = np.where(
        data_max == data_min, np.zeros_like(data_max, dtype=np.float32), data_min
    )

    min_max_normalization_variables = {
        "cn_max": correct_data_max,
        "cn_min": correct_data_min,
    }
    return min_max_normalization_variables


def apply_data_mask(data, mask_variables):
    maksed_data = data * mask_variables["mask"]
    return maksed_data


def apply_covariance_correction(data, covariance, correction_variables):
    covariance = (covariance - correction_variables["min_cov"]) / (
            correction_variables["max_cov"] - correction_variables["min_cov"]
    )
    corrected_data = data - covariance @ correction_variables["beta"]
    return corrected_data


def apply_data_normalization(data, normalization_variables):
    # [-1, 1]
    normalized_data = (data - normalization_variables["cn_mean"]) / (
        normalization_variables["cn_std"]
    )
    return normalized_data


def apply_data_min_max_normalization(data, min_max_normalization_variables):
    # [0, 1]
    min_max_normalized_data = (data - min_max_normalization_variables["cn_min"]) / (
            min_max_normalization_variables["cn_max"]
            - min_max_normalization_variables["cn_min"]
    )
    return min_max_normalized_data


def uniform(data):
    max_value, min_value = np.max(data), np.min(data)

    data = (data - min_value) / (max_value - min_value)
    return data


def clamp(data, a_min, b_max):
    return np.clip(data, a_min, b_max)
