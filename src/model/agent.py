# %% IMPORTS

import numpy as np

from model.visual_processing import SaliencyMap


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
    probabilities = np.maximum(0, flattened_array - flattened_array.max() * precision)
    probabilities /= probabilities.sum()

    sampled_flat_index = rng.choice(a=flattened_array.size, p=probabilities)
    sampled_index = np.unravel_index(sampled_flat_index, array.shape, order="F")

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
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
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

    def __init__(self, environment, focus_params, seed=None):
        """
        Initialize the Agent.

        Args:
            environment: The environment in which the agent operates.
            focus_params: An object containing parameters that define the
                attentional focus, including:
                - sampling_precision: Precision of sampling within the focus.
                - attention_max_variance: Maximum variance allowed for
                  attention.
                - attention_fixed_variance_prop: Proportion of variance that
                  is fixed.
                - attention_center_distance_variance_prop: Proportion of
                  variance based on distance from the center.
                - attention_center_distance_slope: Slope affecting variance
                  based on center distance.
            seed (int, optional): Seed for the random number generator.
                Defaults to 0 if not provided.
        """

        seed = seed or 0
        self.rng = np.random.RandomState(seed)

        self.environment = environment
        self.saliency_mapper = SaliencyMap(focus_params)
        self.sampling_precision = focus_params.agent_sampling_precision
        self.env_height, self.env_width = environment.observation_space["RETINA"].shape[
            :-1
        ]
        self.vertical_variance = focus_params.attention_max_variance * self.env_height
        self.horizontal_variance = focus_params.attention_max_variance * self.env_width
        self.attentional_mask = None
        self.MAX_VARIANCE = focus_params.attention_max_variance
        self.FIXED_VARIANCE_PROP = focus_params.attention_fixed_variance_prop
        self.CENTER_DISTANCE_VARIANCE_PROP = (
            focus_params.attention_center_distance_variance_prop
        )
        self.CENTER_DISTANCE_SLOPE = focus_params.attention_center_distance_slope

        self.params = None

    def set_parameters(self, params=None):
        """
        Set the parameters for the attentional mask.

        This method configures the attentional mask by setting its parameters
        based on the provided coordinates. The mask focuses on a specific area
        of the environment, modulating its amplitude according to the distance
        from the center of the retina.

        Args:
            params (list or array-like, optional): A pair of coordinates
                defining the center of the attentional focus. The coordinates
                should be in a normalized range [0, 1]. If `None`, the
                attentional mask defaults to a uniform distribution. This
                parameter allows modulation of the amplitude of the radial
                focus based on the distance from the center of the retina.
        """

        if params is not None:
            # Ensure parameters are within the valid range and reshape them
            params = np.clip(params, 0, 1).reshape(-1)

            # Store a copy of the parameters
            self.params = np.copy(params)

            # Calculate the environment size
            env_size = np.array([self.env_height, self.env_width])

            # Calculate the scale of the variance based on the distance from
            # the center of the retina
            center = 0.5
            scale = self.MAX_VARIANCE * (
                self.FIXED_VARIANCE_PROP
                + self.CENTER_DISTANCE_VARIANCE_PROP
                * (
                    1
                    - np.tanh(
                        self.CENTER_DISTANCE_SLOPE * np.linalg.norm(params - center)
                    )
                )
            )

            # Adjust parameters to the environment size
            params *= env_size

            # Create the attentional mask using a Gaussian distribution
            self.attentional_mask = gaussian_mask(
                (self.env_height, self.env_width),
                params,
                self.vertical_variance * scale,
                self.horizontal_variance * scale,
                angle=0,
            )
        else:
            # Default to a uniform distribution if no parameters are provided
            self.attentional_mask = np.ones([self.env_height, self.env_width])

    def get_action(self, observation, get_probs=False):
        """Determine the action to take based on the provided observation.

        Args:
        - observation (dict): A dictionary representing the current state of
          the environment. Must contain a key 'RETINA' which provides the
          necessary visual input data.
        - get_probs (bool, optional): If True, return probabilities of
          selection.

        Returns:
        - tuple: A tuple containing the action to take, the generated saliency
          map, and the selected salient point. If `get_probs` is True, also
          returns the probabilities.
        """
        retina_image = observation["RETINA"]

        rgb, brightness, adjusted_response = self.saliency_mapper(retina_image)
        color_saliency, _, saliency_map = rgb, brightness, adjusted_response
        saliency_map_adapted = saliency_map.mean(-1)
        mx = saliency_map_adapted.max()
        saliency_map_adapted += mx * 0.01 if mx > 0 else 0.01
        saliency_map_adapted /= mx
        # ascii_imshow(saliency_map_adapted, 10, 10)
        if self.attentional_mask is None:
            self.attentional_mask = np.ones_like(saliency_map_adapted)

        saliency_map_adapted *= self.attentional_mask

        salient_point, probabilities = sampling(
            saliency_map_adapted, self.sampling_precision, self.rng
        )

        normalized_action = salient_point / self.environment.retina_size

        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (normalized_action - 0.5) * self.environment.retina_scale

        fovea_shape = observation["FOVEA"].shape
        retina_shape = color_saliency.shape
        start = retina_shape[0] // 2 - fovea_shape[0] // 2
        end = start + fovea_shape[0]
        fovea = color_saliency[start:end, start:end, :]

        if get_probs:
            return (
                centered_action,
                saliency_map_adapted,
                probabilities,
                salient_point,
                fovea,
            )
        else:
            return (
                centered_action,
                saliency_map_adapted,
                salient_point,
                fovea,
            )
