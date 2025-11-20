# %% IMPORTS

import numpy as np
from scipy.special import softmax
from scipy.signal import convolve2d


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
    probabilities = np.maximum(
        0, flattened_array - flattened_array.max() * precision
    )
    probabilities /= probabilities.sum()

    sampled_flat_index = rng.choice(a=flattened_array.size, p=probabilities)
    sampled_index = np.unravel_index(
        sampled_flat_index, array.shape, order="F"
    )

    return sampled_index, probabilities


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
        sampling_precision=0.07,
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
        - sampling_precision (float): Precision for sampling ( [0,1[ ).
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
        self.saliency_mapper = SaliencyMap()
        self.sampling_precision = sampling_precision
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

    def get_action(self, observation, get_probs=False):
        """Determine the action to take based on the provided observation.

        Args:
        - observation (dict): A dictionary representing the current state of
          the environment.  Must contain a key 'RETINA' which provides the
          necessary visual input data.

        Returns:
        - tuple: A tuple containing the action to take, the generated saliency
          map, and the selected salient point."""
        retina_image = observation["RETINA"].mean(-1) / 255
        inverted_retina = 1 - retina_image

        saliency_map = self.saliency_mapper(inverted_retina)
        if self.attentional_mask is None:
            self.attentional_mask = np.ones_like(saliency_map)
        saliency_map_adapted = saliency_map

        # border_filter = np.ones_like(saliency_map_adapted)
        # border_filter = scipy.signal.convolve2d(
        #     border_filter,
        #     np.ones([5, 5]) / (5 * 5),
        #     mode="same",
        # )
        # border_filter = np.exp(-(0.1**-2) * (1 - border_filter) ** 2)
        saliency_map_adapted += 1e-5
        # saliency_map_adapted *= border_filter

        saliency_map_adapted *= self.attentional_mask
        # saliency_map_adapted += self.attentional_mask

        salient_point, probabilities = sampling(
            saliency_map_adapted, self.sampling_precision, self.rng
        )

        normalized_action = salient_point / self.environment.retina_size

        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (
            normalized_action - 0.5
        ) * self.environment.retina_scale

        if get_probs:
            return (
                centered_action,
                saliency_map_adapted,
                probabilities,
                salient_point,
            )
        else:
            return centered_action, saliency_map_adapted, salient_point
