"""Saliency-driven agent with a Gaussian attentional mask.

The agent filters the retina with the channel-opponent Gabor bank of
model.visual_processing.SaliencyMap, weights the resulting saliency by an
attentional mask centered on a normalized point, samples a salient pixel,
and returns the eye displacement that centers the retina on it together
with the color-saliency fovea used as visual input by the maps.
"""
# %% IMPORTS

import cv2
import numpy as np

from model.visual_processing import SaliencyMap


# %% SAMPLE FUNCTION
def sampling(array, precision=0.8, rng=None):
    """
    Sample an index from the array with probability proportional to the
    part of each value exceeding a fraction of the maximum.

    p(i) is proportional to max(0, a_i - precision * max(a)); if no value
    exceeds the threshold, sampling is uniform.

    Args:
    - array (np.ndarray): 2D array (H, W) from which to sample.
    - precision (float): Threshold as a fraction of the maximum, in [0, 1);
      default is 0.8.
    - rng (np.random.RandomState): The random number generator; a new
      RandomState(0) if None.

    Returns:
    - tuple: (sampled_index, probabilities). sampled_index is the (row, col)
      index of the sampled element; probabilities is the flat (H*W,)
      C-order distribution.
    """

    rng = rng or np.random.RandomState(0)

    flattened_array = array.flatten()

    probabilities = np.maximum(0, flattened_array - flattened_array.max() * precision)

    sm = probabilities.sum()
    probabilities /= sm if sm > 0 else len(probabilities)
    probabilities.fill(1 / len(probabilities)) if sm <= 0 else None

    sampled_flat_index = rng.choice(a=flattened_array.size, p=probabilities)
    sampled_index = np.unravel_index(sampled_flat_index, array.shape)

    return sampled_index, probabilities


def gaussian_mask(shape, mean, v1, v2, angle):
    """
    Generate a 2D Gaussian mask with a specified shape, mean, variances,
    and rotation angle.

    Parameters:
    shape (tuple): Dimensions of the gaussian mask (height, width).
    mean (array-like): The mean of the Gaussian distribution in pixels,
        ordered as (x, y) = (column, row).
    v1 (float): Variance along x (columns).
    v2 (float): Variance along y (rows).
    angle (float): Rotation angle of the Gaussian distribution in radians.

    Returns:
    numpy.ndarray: An unnormalized 2D Gaussian mask of the specified shape,
        peak value 1.
    """

    # Generate data points
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    x = np.column_stack([cols.ravel(), rows.ravel()])

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
                attentional focus and the saliency filters, including:
                - agent_sampling_precision: sampling threshold (see
                  `sampling`).
                - gabor_*: Gabor bank parameters (see SaliencyMap).
                - test_fovea: use the raw FOVEA instead of the saliency
                  fovea as visual output.
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
        self.focus_params = focus_params
        # Last retina and its saliency, reused when the same retina is
        # filtered again (get_fovea then get_action at a saccade)
        self._cached_retina = None
        self._cached_saliency = None

        self.params = None

    def _saliency(self, retina_image):
        """Gabor saliency of a retina, reusing the last result if the retina
        is unchanged.

        Args:
            retina_image (np.ndarray): (H, W, 3) retina observation.

        Returns:
            tuple: (rgb, brightness, adjusted) from SaliencyMap.
        """
        if self._cached_retina is None or not np.array_equal(
            self._cached_retina, retina_image
        ):
            self._cached_retina = np.copy(retina_image)
            self._cached_saliency = self.saliency_mapper(retina_image)
        return self._cached_saliency

    def set_parameters(self, params=None):
        """
        Set the parameters for the attentional mask.

        This method configures the attentional mask by setting its parameters
        based on the provided coordinates. The mask focuses on a specific area
        of the environment, modulating its amplitude according to the distance
        from the center of the retina.

        Args:
            params (list or array-like, optional): A pair of coordinates
                defining the center of the attentional focus, as image
                (x, y) = (column, row) normalized to [0, 1]. If `None`, the
                attentional mask defaults to a uniform distribution. This
                parameter allows modulation of the amplitude of the radial
                focus based on the distance from the center of the retina.
        """

        if params is not None:
            # Ensure parameters are within the valid range and reshape them
            params = np.clip(params, 0, 1).reshape(-1)

            # Store a copy of the parameters
            self.params = np.copy(params)

            # Calculate the environment size as (x, y)
            env_size = np.array([self.env_width, self.env_height])

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
                self.horizontal_variance * scale,
                self.vertical_variance * scale,
                angle=0,
            )
        else:
            # Default to a uniform distribution if no parameters are provided
            self.attentional_mask = np.ones([self.env_height, self.env_width])

    def _fovea(self, observation, color_saliency):
        """Crop, resize and scale the color saliency into the maps' input.

        Args:
            observation (dict): environment observation (its FOVEA is
                returned if test_fovea).
            color_saliency (np.ndarray): (H, W, 3) color-opponent saliency
                of the retina.

        Returns:
            np.ndarray: (*fovea_size, 3) fovea.
        """
        if self.focus_params.test_fovea:
            return observation["FOVEA"]
        fovea_height, fovea_width = self.environment.fovea_size
        fovea_scale = np.asarray(self.environment.fovea_scale)
        start = np.asarray(self.environment.retina_size) // 2 - fovea_scale // 2
        end = start + fovea_scale
        crop = color_saliency[start[0] : end[0], start[1] : end[1], :]
        # cv2 takes the target size as (width, height)
        fovea = cv2.resize(crop, (int(fovea_width), int(fovea_height)))
        fovea *= self.focus_params.fovea_gain
        return fovea

    def get_fovea(self, observation):
        """Return the visual input of the maps for an observation.

        This is the same fovea returned by get_action and recorded during
        training, so it must be used whenever the controller is queried.

        Args:
            observation (dict): environment observation with "RETINA" (and
                "FOVEA" if test_fovea).

        Returns:
            np.ndarray: (*fovea_size, 3) color-saliency fovea (x
            fovea_gain), or the raw FOVEA if test_fovea.
        """
        if self.focus_params.test_fovea:
            return observation["FOVEA"]
        color_saliency, _, _ = self._saliency(observation["RETINA"])
        return self._fovea(observation, color_saliency)

    def get_action(self, observation, get_probs=False):
        """Determine the action to take based on the provided observation.

        Args:
        - observation (dict): A dictionary representing the current state of
          the environment. Must contain a key 'RETINA' which provides the
          necessary visual input data.
        - get_probs (bool, optional): If True, return probabilities of
          selection.

        Returns:
        - tuple: (action, saliency map, salient point, fovea), or
          (action, saliency map, probabilities, salient point, fovea) if
          `get_probs` is True.
          - action (np.ndarray, (2,)): retina displacement in task-space
            units, in [-retina_scale/2, retina_scale/2], y axis pointing up.
          - saliency map (np.ndarray, (H, W)): channel-mean adjusted
            saliency, divided by its maximum, weighted by the attentional
            mask.
          - salient point (np.ndarray, (2,)): sampled pixel as (x, y) =
            (column, row).
          - fovea (np.ndarray, (*fovea_size, 3)): the color saliency in the
            central fovea_scale retina pixels, resized to fovea_size and
            multiplied by fovea_gain (the raw FOVEA observation if
            test_fovea).
        """
        retina_image = observation["RETINA"]

        rgb, brightness, adjusted_response = self._saliency(retina_image)
        color_saliency, _, saliency_map = rgb, brightness, adjusted_response
        saliency_map_adapted = saliency_map.mean(-1)
        mx = saliency_map_adapted.max()
        saliency_map_adapted += mx * 0.01 if mx > 0 else 0.01
        # A blank retina has mx == 0: the offset alone then gives a
        # uniform map
        saliency_map_adapted /= mx if mx > 0 else saliency_map_adapted.max()
        if self.attentional_mask is None:
            self.attentional_mask = np.ones_like(saliency_map_adapted)

        saliency_map_adapted *= self.attentional_mask

        (row, col), probabilities = sampling(
            saliency_map_adapted, self.sampling_precision, self.rng
        )
        salient_point = np.array([col, row])

        normalized_action = salient_point / np.array(
            [self.env_width, self.env_height]
        )

        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (normalized_action - 0.5) * self.environment.retina_scale

        fovea = self._fovea(observation, color_saliency)

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
