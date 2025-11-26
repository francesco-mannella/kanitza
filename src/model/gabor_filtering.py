# %%

import io
import os
import urllib

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


# %% GABOR FILTER FUNCTION
def gabor_kernel(frequency, orientation, sigma, sigma_y=None, phase_offset=0, size=5):
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
    x_grid, y_grid = np.ogrid[-half_size : (half_size + 1), -half_size : (half_size + 1)]
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


class ChannelGaborFilter:
    """
    Apply multi-scale, multi-orientation channel-opponent Gabor filters.

    This class processes an RGB image by applying Gabor filters at various
    scales and orientations, using channel-opponent and uniform masks for
    comparative color feature detection. The image is filtered through masks
    that isolate and contrast the red, green, and blue channels, as well as
    provide uniform and inverted uniform responses, enabling assessment of
    color- and brightness-dependent spatial features.

    Output channels:
        - Channel 0: Salience of red features relative to green.
        - Channel 1: Salience of green features relative to red.
        - Channel 2: Salience of blue features relative to red and green.
        - Channel 3: Composite channel representing overall brightness
          (uniform response to all channels, normalized).

    Args:
        scale_list (list[float]): List of scales (sigma values) for Gabor
            kernels.
        orientation_list (list[float]): List of filter orientations in radians.
        frequency (float): Spatial frequency parameter for Gabor kernels.
        phase_offset (float): Phase offset for Gabor kernels, in radians.
        kernel_size (int): Width/height of square Gabor kernels.
        filter_slope (float): Exponential slope for feature nonlinearity.
        sigma_y_multiplier (float): Elongation factor for the kernel y-axis.
        rgb_prop (float): Proportion for RGB in adjusted output.
        bright_prop (float): Proportion for brightness in adjusted output.
    """

    def __init__(
        self,
        scale_list,
        orientation_list,
        frequency,
        phase_offset,
        kernel_size=21,
        filter_slope=0.8,
        sigma_y_multiplier=5,
        rgb_prop=0.7,
        bright_prop=0.3,
    ):
        """Initialize ChannelGaborFilter with filter and mask parameters.

        Args:
            scale_list (list[float]): List of Gabor kernel scales (sigmas).
            orientation_list (list[float]): List of orientations (radians).
            frequency (float): Gabor kernel frequency.
            phase_offset (float): Gabor kernel phase offset (radians).
            kernel_size (int): Size of square Gabor kernel.
            filter_slope (float): Slope for nonlinearity.
            sigma_y_multiplier (float): Y-axis elongation factor.
            rgb_prop (float): Proportion for RGB in adjusted output.
            bright_prop (float): Proportion for brightness in adjusted output.
        """
        self.scale_list = scale_list
        self.orientation_list = orientation_list
        self.frequency = frequency
        self.phase_offset = phase_offset
        self.kernel_size = kernel_size
        self.filter_slope = filter_slope
        self.sigma_y_multiplier = sigma_y_multiplier
        self.rgb_prop = rgb_prop
        self.bright_prop = bright_prop
        self.mask_channel_weights = [
            (1.0, -1.0, 0),  # Red vs Green
            (-1.0, 1.0, 0),  # Green vs Red
            (0, -1.0, 1.0),  # Blue vs Green
            (0.0, 1.0, -1.0),  # Green vs blue
            (-1.0, 0, 1.0),  # Blue vs Red
            (1.0, 0, -1.0),  # Red vs blue
            (0.3, 0.3, 0.3),  # Uniform (brightness)
            (-0.3, -0.3, -0.3),  # Inverted uniform
        ]

    def __call__(self, image):
        """Apply multi-scale, multi-orientation channel-opponent Gabor filters.

        This function processes an RGB image by applying Gabor filters at
        various scales and orientations, using channel-opponent and uniform
        masks for comparative color feature detection. The image is projected
        onto each mask, nonlinearly enhanced, and filtered with Gabor kernels.
        Filtered responses are accumulated into color-opponent and brightness
        channels. The output includes filtered RGB channels, a brightness
        channel, and an adjusted RGB output combining both.

        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Filtered RGB channels of shape (H, W, 3),
                  normalized to [0, 1].
                - np.ndarray: Brightness channel of shape (H, W), normalized
                  to [0, 1].
                - np.ndarray: Adjusted RGB output of shape (H, W, 3),
                  combining filtered RGB and brightness, normalized to [0, 1].
        """

        # Ensure the image has at most 3 channels (RGB)
        if image.shape[-1] > 3:
            image = image[:, :, :3]

        # Extract image dimensions
        h, w, c = image.shape

        # Initialize output array with an extra channel for brightness
        output = np.zeros((h, w, c + 1))

        # Iterate over each set of mask channel weights
        for weights in self.mask_channel_weights:
            # Apply mask to the image
            masked = image @ weights

            # Apply Gaussian-like transformation
            # masked = np.exp(-(self.filter_slope**-2) * (masked - 1) ** 2)

            # Iterate over scales and orientations for Gabor filtering
            for sigma in self.scale_list:
                for theta in self.orientation_list:
                    # Create Gabor kernel
                    kernel = gabor_kernel(
                        size=self.kernel_size,
                        sigma=sigma,
                        sigma_y=sigma * self.sigma_y_multiplier,
                        orientation=theta,
                        frequency=self.frequency,
                        phase_offset=self.phase_offset,
                    )

                    # Apply convolution and take absolute value
                    filtered = np.abs(convolve(masked, kernel, mode="nearest"))

                    # Distribute filtered results into output channels
                    if not all(x == weights[0] for x in weights):
                        output[:, :, np.argmax(weights)] += filtered
                    else:
                        output[:, :, -1] += filtered

        # Normalize the output to the range [0, 1]
        output = (output - output.min()) / (output.max() - output.min())

        # Separate RGB and brightness channels
        filtered_rgb = output[:, :, :3]
        brightness = output[:, :, 3]

        # Expand brightness dimension for broadcasting
        brightness_exp = np.expand_dims(brightness, -1)

        # Adjust RGB values based on brightness
        adjusted_rgb = (filtered_rgb * self.rgb_prop) + (
            brightness_exp * self.bright_prop
        )

        # Return the processed image components
        return filtered_rgb, brightness, adjusted_rgb


if __name__ == "__main__":
    # """
    # Demo: Visualizes the effect of multi-scale, multi-orientation Gabor
    # filtering on a set of images. For each image, applies channel-wise Gabor
    # filters and displays the original, filtered channels, and various
    # visualizations of the filter responses for qualitative analysis.
    # """

    # Collect all jpg image file paths from the specified directory
    # image_files = glob.glob("photos_no_class/*jpg")
    # image_files = glob.glob("base_imgs/*jpg")

    images = [
        # "https://elements-resized.envatousercontent.com/envato-dam-assets-production/EVA/TRX/f0/df/51/9b/a2/v1_E10/E108QOQX.jpg?w=1600&cf_fit=scale-down&mark-alpha=18&mark=https%3A%2F%2Felements-assets.envato.com%2Fstatic%2Fwatermark4.png&q=85&format=auto&s=cf8933d911882d0def266f4f7ecc7111e3834ec380fc4104713c97a270a45902",
        # "https://www.astrofilifiemme.it/wp-content/uploads/2021/04/Jupiter-1536x864.jpg",
        # "https://rare-gallery.com/thumbs/527927-real-nature.jpg",
        # "https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs42003-022-03518-2/MediaObjects/42003_2022_3518_Fig1_HTML.png?as=webp",
        # "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuxI_vIJ5d3iudbuw4kLKrCV3sxzhzebO3RQ&s",
        # "https://slyvi-hosting.slyvi.it/teampages/3176851814145/images/imported/uploads/news/chris-du-plessis-torna-in-campo-con-il-biella-rugby-17829.png",
        # "https://nebraskapublicmedia.org/assets/images/download_-_2025-07-21T112133.272.min-800x600.png",
        f"file://{os.path.dirname(os.path.abspath(__file__))}/gabor_test.png"
    ]

    # Define Gabor fil:ter parameters
    scales = [8]
    orientations = np.pi * np.linspace(0, 360, 10) / 180.0
    frequency = 0.09
    phase_offset = -np.pi * (0.5 - 25e-3)
    kernel_size = 3
    filter_slope = 0.02
    sigma_y_multiplier = 6
    rgb_prop = 1.0
    bright_prop = 1.0
    gabor_manager = ChannelGaborFilter(
        scales,
        orientations,
        frequency,
        phase_offset,
        kernel_size,
        filter_slope=filter_slope,
        sigma_y_multiplier=sigma_y_multiplier,
        rgb_prop=rgb_prop,
        bright_prop=bright_prop,
    )

    # Process each image in the shuffled list
    for image_url in images:

        # Load image and ensure only RGB channels are used
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()

        # Use a BytesIO object to allow seek operations
        image_bytes = io.BytesIO(image_data)

        # Use plt.imread with a file object
        image = plt.imread(image_bytes, format="jpeg")  # Adjust format if necessary

        # Apply channel-wise Gabor filters to the image
        rgb, brightness, adjusted_rgb = gabor_manager(image)

        # Create a 3x4 grid of subplots for visualization
        plt.close("all")
        fig, axes = plt.subplots(3, 4, figsize=(12, 6))
        axes = axes.ravel()

        #####
        # Hide axes for cleaner visualization
        for ax in axes:
            ax.set_axis_off()

        # Display original image and filtered channels
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[1].imshow(rgb[:, :, 0], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[1].set_title("Red Channel")
        axes[2].imshow(rgb[:, :, 1], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[2].set_title("Green Channel")
        axes[3].imshow(rgb[:, :, 2], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[3].set_title("Blue Channel")
        axes[4].imshow(brightness, vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[4].set_title("Brightness")

        # Combine RGB and brightness channels for enhanced visualization
        axes[5].imshow(np.clip(adjusted_rgb, 0, 1))
        axes[5].set_title("Adjusted RGB")

        # Visualize the maximum response across RGB channels plus brightness
        overall_brightness = adjusted_rgb.max(-1)
        axes[6].imshow(
            overall_brightness,
            vmin=0,
            vmax=1,
            cmap=plt.cm.gray,
        )
        axes[6].set_title("Overall Brightness")
        fig.tight_layout(pad=0.2)
        #####

        # Show the figure with all visualizations
        plt.show()
        input("Press any key for next image")
    print("Done")
