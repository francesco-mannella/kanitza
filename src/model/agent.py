# %% IMPORTS

import numpy as np
from scipy.special import softmax


# %% SAMPLE FUNCTION
def sampling(array, precision=0.01, rng=None):
    """
    Sample an index from the array based on probabilities derived from softmax.

    Args:
    - array (np.ndarray): The input array from which to sample.
    - precision (float): A parameter controlling the softness of the softmax;
      default is 0.6.
    - rng (np.random.RandomState): The random number generator

    Returns:
    - tuple: The sampled index in the same shape as the input array.
    """

    rng = rng or np.random.RandomState(0)

    flattened_array = array.flatten()
    probabilities = softmax(flattened_array / precision)
    probabilities[probabilities < probabilities.max() * 0.999] = 0
    probabilities /= probabilities.sum()

    sampled_flat_index = rng.choice(a=flattened_array.size, p=probabilities)
    sampled_index = np.unravel_index(
        sampled_flat_index, array.shape, order="F"
    )

    return sampled_index


def gaussian_mask(shape, mean, v1, v2, angle):
    """
    Generate a 2D Gaussian mask with a specified shape, mean, variances,
    and rotation angle.

    Parameters:
    shape (tuple): Dimensions of the gaussian mask (height, width).
    mean (array-like): The mean of the Gaussian distribution (mean_x, mean_y).
    v1 (float): Variance along the x-axis.
    v2 (float): Variance along the y-axis.
    angle (float): Rotation angle of the Gaussian distribution in radians.

    Returns:
    numpy.ndarray: A 2D Gaussian mask of the specified shape.
    """

    # Generate data points
    tx = np.arange(shape[0])
    ty = np.arange(shape[1])
    tX, tY = np.meshgrid(tx, ty)
    x = np.column_stack([tX.flat, tY.flat])

    # Compute rotated covariance matrix
    cov_matrix = np.array([[v1, 0], [0, v2]])
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_cov_matrix = rot @ cov_matrix @ rot.T

    x_minus_mu = x - mean
    inv_cov = np.linalg.inv(rotated_cov_matrix)

    result = np.einsum("...k,kl,...l->...", x_minus_mu, inv_cov, x_minus_mu)
    return np.exp(-0.5 * result).reshape(*shape)


# %% AGENT CLASS
class Agent:
    """
    Agent that interacts with the environment and determines actions based on
    saliency maps.
    """

    def __init__(
        self,
        environment,
        sampling_threshold=0.07,
        max_variance=1,
        seed=None,
        attention_max_variance=1.0,
        attention_fixed_variance_prop=0.1,
        attention_center_distance_variance_prop=0.9,
        attention_center_distance_slope=3.0,
    ):
        """
        Initialize the Agent.

        Args:
        - environment: The environment.
        - sampling_threshold (float): Threshold for sampling.
        - max_variance (float): Max std of attentional field.
        - seed (int): Seed for the random number generator.
        - attention_max_variance (float): Max variance of attention.
        - attention_fixed_variance_prop (float): Fixed variance prop.
        - attention_center_distance_variance_prop (float): Center dist prop.
        - attention_center_distance_slope (float): Center dist variance slope.
        """

        seed = seed or 0
        self.rng = np.random.RandomState(seed)

        self.environment = environment
        self.sampling_threshold = sampling_threshold
        self.env_height, self.env_width = environment.observation_space[
            "RETINA"
        ].shape[:-1]
        self.vertical_variance = max_variance * self.env_height
        self.horizontal_variance = max_variance * self.env_width
        self.attentional_mask = None

        self.MAX_VARIANCE = attention_max_variance
        self.FIXED_VARIANCE_PROP = attention_fixed_variance_prop
        self.CENTER_DISTANCE_VARIANCE_PROP = (
            attention_center_distance_variance_prop
        )
        self.CENTER_DISTANCE_SLOPE = attention_center_distance_slope

        self.params = None

    def set_parameters(self, params=None):
        """
        Set the parameters for the attentional mask.

        Args:
        - params (list or array-like): The parameters to set for the
          attentional mask.
        """

        if params is not None:

            params = np.clip(params, 0, 1).reshape(-1)

            self.params = np.copy(params)

            env_size = np.array([self.env_height, self.env_width])

            center = 0.5
            scale = self.MAX_VARIANCE * (
                self.FIXED_VARIANCE_PROP
                + self.CENTER_DISTANCE_VARIANCE_PROP
                * (
                    1
                    - np.tanh(
                        self.CENTER_DISTANCE_SLOPE
                        * np.linalg.norm(params - center)
                    )
                )
            )

            params *= env_size

            self.attentional_mask = gaussian_mask(
                (self.env_height, self.env_width),
                params,
                self.vertical_variance * scale,
                self.horizontal_variance * scale,
                angle=0,
            )
        else:
            self.attentional_mask = np.ones([self.env_height, self.env_width])

    def get_attentional_map_and_point(self, saliency_map):
        saliency_map = saliency_map.max(-1)
        if self.attentional_mask is None:
            self.attentional_mask = np.ones_like(saliency_map)
        saliency_map_adapted = saliency_map
        saliency_map_adapted *= self.attentional_mask

        salient_point = sampling(
            saliency_map_adapted, self.sampling_threshold, self.rng
        )
        return saliency_map_adapted, salient_point

    def get_action(self, saliency):
        """Calculates and returns the next action based on the given saliency.

        This method processes the input saliency to generate an attentional map,
        selects a salient point, and computes a normalized and centered action
        for the environment.

        Args:
            saliency (dict): Dictionary representing the current state of the
                environment, typically containing saliency information.

        Returns:
            tuple: A tuple containing:
                - centered_action (np.ndarray): The normalized and centered
                action to take.
                - attentional_map (np.ndarray): The generated saliency map.
                - attention_point (np.ndarray): The selected salient point.
        """
        attentional_map, attention_point = self.get_attentional_map_and_point(
            saliency
        )

        normalized_action = attention_point / self.environment.retina_size
        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (
            normalized_action - 0.5
        ) * self.environment.retina_scale

        return centered_action, attentional_map, attention_point
