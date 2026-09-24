"""Saliency front end: the channel-opponent Gabor bank built from Parameters."""
import numpy as np

from model.gabor_filtering import ChannelGaborFilter


class SaliencyMap:
    """
    Generates a saliency map using Gabor filters.

    Wraps model.gabor_filtering.ChannelGaborFilter configured from the
    gabor_* parameters. Orientations are linspace(0, 180,
    gabor_orientation_bins)[:-1] degrees: bins - 1 orientations over the
    half circle (theta and theta + 180 give nearly the same absolute
    response with these near-odd kernels).
    """

    def __init__(self, params):
        """
        Args:
            params (Parameters): provides gabor_scales,
                gabor_orientation_bins, gabor_frequency, gabor_phase_offset,
                gabor_kernel_size, gabor_sigma_y_multiplier, gabor_rgb_prop
                and gabor_bright_prop.
        """
        scales = params.gabor_scales
        orientation_bins = params.gabor_orientation_bins
        frequency = params.gabor_frequency
        phase_offset = params.gabor_phase_offset
        kernel_size = params.gabor_kernel_size
        sigma_y_multiplier = params.gabor_sigma_y_multiplier
        rgb_prop = params.gabor_rgb_prop
        bright_prop = params.gabor_bright_prop
        orientations = np.pi * np.linspace(0, 180, orientation_bins)[:-1] / 180.0
        self.gabor_manager = ChannelGaborFilter(
            scales,
            orientations,
            frequency,
            phase_offset,
            kernel_size,
            sigma_y_multiplier=sigma_y_multiplier,
            rgb_prop=rgb_prop,
            bright_prop=bright_prop,
        )

    def __call__(self, input_image):
        """
        Apply the Gabor filters to the input image to generate the saliency
        map.

        Args:
        - input_image (np.ndarray): (H, W, 3) RGB image, uint8 in [0, 255]
          or float in [0, 1] (rescaled by 1/255 if its maximum exceeds 1).

        Returns:
        - tuple: (rgb, brightness, adjusted) as returned by
          ChannelGaborFilter: (H, W, 3) color-opponent responses and
          (H, W) brightness response, jointly normalized to [0, 1], and
          the (H, W, 3) mix rgb * gabor_rgb_prop + brightness *
          gabor_bright_prop.
        """

        input_image = input_image.astype(float)
        if input_image.max() > 1:
            input_image /= 255.0

        rgb, brightness, adjusted_response = self.gabor_manager(input_image)

        return rgb, brightness, adjusted_response
