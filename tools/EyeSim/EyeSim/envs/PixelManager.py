import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


class PathToPixelsConverter:
    """Converts a path defined by vertices to a pixelated image."""

    def __init__(self, dims, shape, radius):
        """Initializes the converter.

        Args:
            dims (list): The original dimensions of the image.
            shape (list): The shape of the image (height, width).
            radius (float): Tolerance (task-space units) used when testing
                whether a pixel center lies inside a path.
        """
        self.shape = list(shape)
        self.scale = dims/self.shape
        self.n_pixels = self.shape[0] * self.shape[1]
        self.displace = np.zeros(2)

        # Create x and y coordinate arrays
        self.x = (
            np.arange(
                -self.shape[0] // 2,
                self.shape[0] // 2,
            )
            + 1
        )
        self.y = (
            np.arange(
                -self.shape[1] // 2,
                self.shape[1] // 2,
            )
            + 1
        )
        # Create a grid of coordinates
        X, Y = np.meshgrid(self.x, self.y[::-1])
        self.grid = np.vstack((X.flatten(), Y.flatten())).T
        self.radius = radius

    def set_displace(self, displace):
        """Set the task-space position of the image center."""
        self.displace *= 0
        self.displace += displace

    def path2pixels(self, vertices):
        """Converts a path to a pixel image (grayscale).

        Args:
            vertices (np.ndarray): Vertices defining the path.

        Returns:
            np.ndarray: A pixel image representing the path.
        """
        points = self.grid * self.scale + self.displace

        path = Path(vertices)
        points_in_path = path.contains_points(points, radius=self.radius)
        img = 2.0 * points_in_path.reshape(*self.shape, order="F").T - 1
        return img

    def path_mask(self, vertices, points=None):
        """Boolean image of the pixels whose centers lie inside a path.

        Args:
            vertices (np.ndarray): Vertices defining the path.
            points (np.ndarray, optional): (n_pixels, 2) task-space pixel
                centers; computed from the current displacement if None.

        Returns:
            np.ndarray: (shape[1], shape[0]) boolean mask.
        """
        if points is None:
            points = self.grid * self.scale + self.displace

        path = Path(vertices)
        points_in_path = path.contains_points(points, radius=self.radius)
        return points_in_path.reshape(*self.shape, order="F").T

    def path2pixels_color(self, vertices, color):
        """Converts a path to a colored pixel image.

        Args:
            vertices (np.ndarray): Vertices defining the path.
            color (list): RGB color of the path.

        Returns:
            np.ndarray: A colored pixel image representing the path.
        """
        return self.path_mask(vertices)[:, :, None] * np.array(color[:3])

    def merge_imgs(self, vertices_list, colors, zorder, background=(1, 1, 1)):
        """Merges multiple paths into a single colored image.

        Args:
            vertices_list (list): List of vertices for each path.
            colors (list): List of RGB colors for each path.
            zorder (tuple): Tuple defining the drawing order.
            background: background color

        Returns:
            np.ndarray: (H, W, 3) float image in [0, 1]. Where paths overlap,
            the one with the lowest zorder wins.
        """
        img = np.zeros((*self.shape, 3))
        mask = np.zeros(self.shape, dtype=bool)
        points = self.grid * self.scale + self.displace
        # Paths entirely outside this box (pixel centers +- the inclusion
        # tolerance) cannot contain any pixel center
        margin = abs(self.radius)
        view_min = points.min(0) - margin
        view_max = points.max(0) + margin

        # Sort vertices and colors based on zorder
        sorted_indices = sorted(range(len(zorder)), key=lambda k: zorder[k])
        sorted_vertices_list = [vertices_list[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        # Draw background first
        img[:] = background

        # Iterate through sorted lists to draw in correct order
        for vertices, color in zip(sorted_vertices_list, sorted_colors):
            vertices = np.asarray(vertices)
            if np.any(vertices.max(0) < view_min) or np.any(vertices.min(0) > view_max):
                continue
            current_mask = self.path_mask(vertices, points)
            # Draw only where not already drawn
            new = current_mask & ~mask
            img[new] = np.asarray(color[:3], dtype=float)
            # Update mask with current object
            mask = mask | current_mask

        # Add new vertex with background color and highest z-order
        background_vertices = np.array(
            [
                [0, 0],
                [0, self.shape[1]],
                [self.shape[0], self.shape[1]],
                [self.shape[0], 0],
            ]
        )
        current_mask = self.path_mask(background_vertices, points)
        img[current_mask & ~mask] = np.asarray(background, dtype=float)
        mask = mask | current_mask

        return img


if __name__ == "__main__":
    scale = [30, 30]
    shape = [64, 64]
    radius = 0.0
    converter = PathToPixelsConverter(scale, shape, radius)

    # Define a triangle path
    vertices = np.array([[0, 0], [1, 1], [0, 1]]) * 5

    # Convert the path to pixels
    img = converter.path2pixels(vertices)

    # Display the image (requires matplotlib)

    plt.imshow(
        img,
        extent=[
            converter.x[0],
            converter.x[-1],
            converter.y[0],
            converter.y[-1],
        ],
    )
    plt.show()

    # New demo code for path2pixels_color
    color = [1, 0, 0]  # Red color
    img_color = converter.path2pixels_color(vertices, color)

    plt.imshow(
        img_color,
        extent=[
            converter.x[0],
            converter.x[-1],
            converter.y[0],
            converter.y[-1],
        ],
    )
    plt.show()

    vertices_list = [
        np.array([[0, 0], [1, 1], [0, 1]]) * 5,  # Triangle
        np.array([[0, 0], [-1, -1], [0, -1]]) * 5,  # Another triangle
        np.array([[0, -12], [-4, 8], [12, 3]]),  # Square
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]  # Red, Green, Blue, White

    vertices_list = vertices_list
    zorder = (0, 1, 2)

    merged_img = converter.merge_imgs(vertices_list, colors, zorder)

    plt.imshow(
        merged_img,
        extent=[
            converter.x[0],
            converter.x[-1],
            converter.y[0],
            converter.y[-1],
        ],
    )
    plt.show()
