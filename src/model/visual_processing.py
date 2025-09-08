import numpy as np

from model.gabor_filtering import ChannelGaborFilter


class SaliencyMap:
    """
    Generates a saliency map using Gabor filters.
    """

    def __init__(self, params):
        scales = params.gabor_scales
        orientation_bins = params.gabor_orientation_bins
        frequency = params.gabor_frequency
        phase_offset = params.gabor_phase_offset
        kernel_size = params.gabor_kernel_size
        filter_slope = params.gabor_filter_slope
        sigma_y_multiplier = params.gabor_sigma_y_multiplier
        bw_channel_ratio = params.gabor_bw_channel_ratio
        rgb_prop = params.gabor_rgb_prop
        bright_prop = params.gabor_bright_prop
        orientations = np.pi * np.linspace(0, 360, orientation_bins) / 180.0
        self.gabor_manager = ChannelGaborFilter(
            scales,
            orientations,
            frequency,
            phase_offset,
            kernel_size,
            filter_slope=filter_slope,
            sigma_y_multiplier=sigma_y_multiplier,
            bw_channel_ratio=bw_channel_ratio,
            rgb_prop=rgb_prop,
            bright_prop=bright_prop,
        )

    def __call__(self, input_image):
        """
        Apply the Gabor filters to the input image to generate the saliency
        map.

        Args:
        - input_image (np.ndarray): The input image.

        Returns:
        - np.ndarray: The generated saliency map.
        """

        input_image = input_image.astype(float)
        if input_image.max() > 1:
            input_image /= 255.0
        rgb, brightness, adjusted_response = self.gabor_manager(input_image)

        return rgb, brightness, adjusted_response
