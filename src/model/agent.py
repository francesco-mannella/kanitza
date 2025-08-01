# %% IMPORTS

import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from scipy.special import softmax


# %% GABOR FILTER FUNCTION
def gabor_filter(
    frequency, orientation, sigma, sigma_y=None, phase_offset=0, size=5
):
    """
    Generate a Gabor filter.

    Args:
    - frequency (float): Spatial frequency of the harmonics.
    - orientation (float): Orientation of the Gabor filter in radians.
    - sigma (float): Standard deviation of the Gaussian envelope.
    - sigma_y (float): Standard deviation of the Gaussian envelope in the 2nd
      dimension.
    - phase_offset (float): Phase offset of the sine wave.
    - size (int): Size of the filter.

    Returns:
    - np.ndarray: The generated Gabor filter.
    """

    if sigma_y is None:
        sigma_y = sigma

    half_size = size // 2
    x_grid, y_grid = np.ogrid[
        -half_size : (half_size + 1), -half_size : (half_size + 1)
    ]
    rotated_x = x_grid * np.cos(orientation) + y_grid * np.sin(orientation)
    rotated_y = -x_grid * np.sin(orientation) + y_grid * np.cos(orientation)
    gabor = np.exp(
        -(rotated_x**2 / (2 * sigma**2) + rotated_y**2 / (2 * sigma_y**2))
    ) * np.cos(2 * np.pi * frequency * rotated_x + phase_offset)

    # Normalize the Gabor filter by dividing it by the sum of its non-negative
    # elements
    non_negative_sum = np.sum(np.maximum(gabor, 0))
    gabor /= non_negative_sum

    return gabor


# %% SALIENCY MAP CLASS
class SaliencyMap:
    """
    Generates a saliency map using Gabor filters.
    """

    def __init__(self):
        filter_size = 10
        spatial_frequency = 0.1
        gaussian_sigma = 4
        phase_offset = 0
        orientations = np.linspace(0, 1, 9)[:-1] * np.pi

        self.gabor_filters = [
            gabor_filter(
                frequency=spatial_frequency,
                orientation=orientation,
                sigma=gaussian_sigma,
                phase_offset=phase_offset,
                size=filter_size,
            )
            for orientation in orientations
        ]

    def __call__(self, input_image):
        """
        Apply the Gabor filters to the input image to generate the saliency
        map.

        Args:
        - input_image (np.ndarray): The input image.

        Returns:
        - np.ndarray: The generated saliency map.
        """
        accumulated_response = np.zeros_like(input_image)

        for gabor in self.gabor_filters:
            accumulated_response += (
                convolve2d(input_image, gabor, mode="same") / 8
            )

        accumulated_response /= accumulated_response.max() + 1e-4
        clipped_response = np.clip(accumulated_response, 0.4, 1)

        response_range = clipped_response.max() - clipped_response.min()
        if response_range > 1e-30:
            adjusted_response = (
                clipped_response - clipped_response.min()
            ) / response_range
        else:
            adjusted_response = np.zeros_like(clipped_response)

        return adjusted_response


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


class AdaptationManager:

    def __init__(self, rows, cols, decay=0.2):
        self.rows = rows
        self.cols = cols
        self.ue = np.zeros([self.rows, self.cols])
        self.ui = np.zeros([self.rows, self.cols])
        self.decay = decay

    def __call__(self, inp):

        self.ue += self.decay * (inp - self.ui - self.ue)
        self.ui += self.decay * (inp - self.ui)
        return np.maximum(0, self.ue)


def test_adaptation():

    n = 10

    sm = AdaptationManager(n, n)

    inp1 = 1 * (np.random.rand(n, n) > 0.94)
    inp2 = 1 * (np.random.rand(n, n) > 0.94)
    frames = [sm(inp1 if t < 20 else inp2) for t in range(40)]

    fig, ax = matplotlib.pyplot.subplots()
    im = ax.imshow(np.zeros([n, n]), vmin=0, vmax=1)

    def init():
        ax.set_axis_off()
        return (im,)

    def update(frame):
        im.set_array(frame)
        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=50,
    )

    matplotlib.pyplot.show()

    return ani


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
        self.saliency_mapper = SaliencyMap()
        self.sampling_threshold = sampling_threshold
        self.env_height, self.env_width = environment.observation_space[
            "RETINA"
        ].shape[:-1]
        self.vertical_variance = max_variance * self.env_height
        self.horizontal_variance = max_variance * self.env_width
        self.attentional_mask = None
        self.adaptation_manager = AdaptationManager(
            self.env_height, self.env_width
        )
        
        self.MAX_VARIANCE = attention_max_variance
        self.FIXED_VARIANCE_PROP = attention_fixed_variance_prop
        self.CENTER_DISTANCE_VARIANCE_PROP = (
            attention_center_distance_variance_prop
        )
        self.CENTER_DISTANCE_SLOPE = (
            attention_center_distance_slope
        )

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
                * (1 - np.tanh(self.CENTER_DISTANCE_SLOPE * np.linalg.norm(params - center)))
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

    def get_action(self, observation):
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
        saliency_map_adapted = (
            saliency_map  # self.adaptation_manager(saliency_map)
        )
        saliency_map_adapted *= self.attentional_mask

        salient_point = sampling(
            saliency_map_adapted, self.sampling_threshold, self.rng
        )

        normalized_action = salient_point / self.environment.retina_size
        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (
            normalized_action - 0.5
        ) * self.environment.retina_scale

        return centered_action, saliency_map_adapted, salient_point
