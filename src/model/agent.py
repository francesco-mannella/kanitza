#%% IMPORTS
import numpy as np
from scipy.signal import convolve2d
from scipy.special import softmax
import gymnasium as gym
import EyeSim


#%% GABOR FILTER FUNCTION
def gabor_filter(frequency, orientation, sigma, phase_offset, size):
    """
    Generate a Gabor filter.

    Args:
    - frequency (float): Spatial frequency of the harmonics.
    - orientation (float): Orientation of the Gabor filter in radians.
    - sigma (float): Standard deviation of the Gaussian envelope.
    - phase_offset (float): Phase offset of the sine wave.
    - size (int): Size of the filter.

    Returns:
    - np.ndarray: The generated Gabor filter.
    """
    half_size = size // 2
    x_grid, y_grid = np.ogrid[
        -half_size : half_size + 1, -half_size : half_size + 1
    ]
    rotated_x = x_grid * np.cos(orientation) + y_grid * np.sin(orientation)
    rotated_y = -x_grid * np.sin(orientation) + y_grid * np.cos(orientation)
    gabor = np.exp(
        -(rotated_x**2 + rotated_y**2) / (2 * sigma**2)
    ) * np.cos(2 * np.pi * frequency * rotated_x + phase_offset)
    gabor /= np.sum(gabor)
    return gabor


#%% SALIENCY MAP CLASS
class SaliencyMap:
    """
    Generates a saliency map using Gabor filters.
    """

    def __init__(self):
        filter_size = 10
        spatial_frequency = 0.1
        gaussian_sigma = filter_size * 0.5
        phase_offset = 0
        orientations = np.linspace(0, 1, 5)[:-1] * np.pi

        self.gabor_filters = [
            gabor_filter(
                spatial_frequency,
                orientation,
                gaussian_sigma,
                phase_offset,
                filter_size,
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
                convolve2d(input_image, gabor, mode='same') / 4
            )

        adjusted_response = 2 * (np.clip(accumulated_response, 0.2, 1) - 0.5)
        return adjusted_response


#%% SAMPLE FUNCTION
def sampling(array, precision=0.01):
    """
    Sample an index from the array based on probabilities derived from softmax.

    Args:
    - array (np.ndarray): The input array from which to sample.
    - precision (float): A parameter controlling the softness of the softmax;
      default is 0.6.

    Returns:
    - tuple: The sampled index in the same shape as the input array.
    """
    flattened_array = array.flatten()
    probabilities = softmax(flattened_array / precision)
    sampled_flat_index = np.random.choice(
        a=flattened_array.size, p=probabilities
    )
    sampled_index = np.unravel_index(
        sampled_flat_index, array.shape, order='F'
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

    result = np.einsum('...k,kl,...l->...', x_minus_mu, inv_cov, x_minus_mu)
    return np.exp(-0.5 * result).reshape(*shape)


#%% AGENT CLASS
class Agent:
    """
    Agent that interacts with the environment and determines actions based on saliency maps.
    """

    def __init__(self, environment, sampling_threshold=0.07):
        """
        Initialize the Agent with the environment and a saliency mapper.

        Args:
        - environment: The environment in which the agent operates.
        - sampling_threshold (float): The threshold value used in the sampling function. Default is 0.07.
        """
        self.environment = environment
        self.saliency_mapper = SaliencyMap()
        self.sampling_threshold = sampling_threshold
        self.env_height, self.env_width = environment.observation_space[
            'RETINA'
        ].shape[:-1]
        self.vertical_variance = 9 * self.env_height
        self.horizontal_variance = 9 * self.env_width
        self.attentional_mask = None

    def set_parameters(self, params):
        """
        Set the parameters for the attentional mask.

        Args:
        - params (list or array-like): The parameters to set for the attentional mask.
        """
        params = np.clip(params, 0, 1)
        params *= [self.env_height, self.env_width]
        self.attentional_mask = gaussian_mask(
            (self.env_height, self.env_width),
            params,
            self.vertical_variance,
            self.horizontal_variance,
            angle=0,
        )

    def get_action(self, observation):
        """
        Determine the action to take based on the provided observation.

        Args:
        - observation (dict): A dictionary representing the current state of the environment.
          Must contain a key 'RETINA' which provides the necessary visual input data.

        Returns:
        - tuple: A tuple containing the action to take, the generated saliency map, and the selected salient point.
        """
        retina_image = observation['RETINA'].mean(-1) / 255
        inverted_retina = 1 - retina_image

        saliency_map = self.saliency_mapper(inverted_retina)
        if self.attentional_mask is None:
            self.attentional_mask = np.ones_like(saliency_map)

        saliency_map *= self.attentional_mask
        salient_point = sampling(saliency_map, self.sampling_threshold)

        normalized_action = salient_point / self.environment.retina_size
        normalized_action[1] = 1 - normalized_action[1]
        centered_action = (
            normalized_action - 0.5
        ) * self.environment.retina_scale

        return centered_action, saliency_map, salient_point
