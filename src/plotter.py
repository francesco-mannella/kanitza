import EyeSim
import numpy as np
from EyeSim.envs.mkvideo import vidManager
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from params import Parameters


class FoveaPlotter(EyeSim.envs.Simulator.TestPlotter):
    """
    FoveaPlotter is a custom plotter for visualizing the saliency map,
    fovea region, and salient points in an eye simulation environment.
    """

    def __init__(self, env, *args, **kwargs):
        """
        Initializes the FoveaPlotter with a given environment.

        Parameters:
            - env (object): The simulation environment containing retina and
              fovea configurations.
        """
        self.env = env
        self.fig, self.axes = plt.subplots(
            1,
            4,
            figsize=(9, 3),
            gridspec_kw={"width_ratios": np.ones(4), "height_ratios": [1]},
        )
        (self.env_ax, self.fovea_ax, self.saliency_ax, self.filter_ax) = (
            self.axes
        )

        self._initialize_images()
        self._initialize_patches()

        super(FoveaPlotter, self).__init__(
            env, ax=self.env_ax, *args, **kwargs
        )

        for ax in self.axes:
            ax.set_box_aspect(1)

        self._set_titles_and_labels()

    def _initialize_images(self):
        """Initialize image and dot plots for the saliency and fovea views."""
        retina_size = np.zeros(self.env.retina_size)
        self.saliency_image = self.saliency_ax.imshow(
            retina_size, vmin=0, vmax=1
        )
        self.attentional_mask = self.filter_ax.imshow(
            retina_size, vmin=0, vmax=1
        )
        self.highlight_dot = self.saliency_ax.scatter(
            0, 0, c="#ff5555", s=50, alpha=0.5
        )

        fovea_sample = np.zeros_like(
            self.env.observation_space["FOVEA"].sample()
        )
        self.fovea_image = self.fovea_ax.imshow(fovea_sample, vmin=0, vmax=1)

    def _initialize_patches(self):
        """Helper method to initialize the retina and fovea position
        rectangles."""

        pos_retina = self.env.retina_sim_pos - self.env.retina_scale / 2
        self.retina_pos = Rectangle(
            pos_retina, *self.env.retina_scale, lw=0.5, fill=None, color="#888"
        )

        pos_fovea = self.env.retina_sim_pos - self.env.fovea_scale / 2
        self.fovea_pos = Rectangle(pos_fovea, *self.env.fovea_scale, fill=None)

        self.env_ax.add_patch(self.retina_pos)
        self.env_ax.add_patch(self.fovea_pos)

    def _set_titles_and_labels(self):
        """Set titles and labels for each axis in the plot."""
        self.env_ax.set_title("Visual field")
        self.fovea_ax.set_title("Fovea")
        self.saliency_ax.set_title("Retina\n(salience)")
        self.filter_ax.set_title("Retina\n(attentional filter)")
        self.saliency_ax.set_axis_off()
        self.filter_ax.set_axis_off()

    def step(self, saliency_map, salient_point, attentional_mask=None):
        """
        Renders the current state of the environment, updating the saliency
        map, fovea image, and highlights salient points.

        Parameters:
        - saliency_map (np.array): The current saliency map to display.
        - salient_point (tuple): Coordinates of the salient point to highlight.
        - attentional_mask (np.array, optional): The current attentional mask
          of salience, defaults to None.

        """
        self.saliency_image.set_array(saliency_map)
        if attentional_mask is not None:
            self.attentional_mask.set_array(attentional_mask)
        self.highlight_dot.set_offsets(salient_point)

        self.fovea_image.set_array(self.env.observation["FOVEA"])

        self._update_rect_positions()

        self.fig.canvas.draw_idle()
        super(FoveaPlotter, self).step()

    def _update_rect_positions(self):
        """Update positions of retina and fovea rectangles."""
        retina_pos = self.env.retina_sim_pos - self.env.retina_scale / 2
        self.retina_pos.set_xy(retina_pos)

        fovea_pos = self.env.retina_sim_pos - self.env.fovea_scale / 2
        self.fovea_pos.set_xy(fovea_pos)

        self.env_ax.add_patch(self.retina_pos)
        self.env_ax.add_patch(self.fovea_pos)

    def close(self, name=None):
        """Closes the plot and potentially saves an offline video."""
        if self.offline and name:
            self._save_video(name)
        plt.close(self.fig)

    def _save_video(self, name):
        """Save images or videos of the visualizations if offline mode is
        enabled."""

        print(f"Saving {name}.png")
        self.vm.fig.savefig(f"{name}.png", dpi=300)
        print(f"Saving {name}.gif")
        self.vm.mk_video(name=name, dirname=".")


class FakeMapsPlotter:
    """
    A class for a fake map plotter which does not do anything.
    """

    def __init__(self, env, controller, offline=False):
        """Initializes the fake MapsPlotter without any functionality."""
        pass

    def step(self):
        """A placeholder method with no functionality."""
        pass

    def close(self, name=None):
        """A placeholder method with no functionality."""
        pass

    def reshape_fovea_weights(self, weights):
        """A placeholder method always returns None."""
        return None


class MapsPlotter:
    """
    A class for plotting fovea and attentional maps using matplotlib.
    """

    def __init__(
        self, env, controller, offline=False, video_frame_duration=200
    ):
        """
        Initializes the MapsPlotter with the given controller and environment.

        Parameters:
        - env (object): The environment within which the controller operates.
        - controller (object): Manages the weights to be plotted.
        - offline (bool): Indicates if operations should be done offline.
        - video_frame_duration (int): Duration for video frames in milliseconds.
        """
        self.controller = controller
        self.env = env
        self.env_height, self.env_width = env.observation_space[
            "RETINA"
        ].shape[:-1]
        self.offline = offline
        self.params = Parameters()

        self.side = int(np.sqrt(self.params.maps_output_size))
        self.fovea_size = env.fovea_size[0]

        self.palette = self._create_palette()
        self.att_palette = self._create_palette(T=True)

        self.fig, (
            self.visual_conditions_map_ax,
            self.visual_effects_map_ax,
            self.attention_map_ax,
        ) = plt.subplots(1, 3, figsize=(9, 3))

        self.vm = vidManager(
            self.fig, name="maps", dirname=".", duration=video_frame_duration
        )
        self.grid = self._prepare_grid()
        self.saccade = None

        self._initialize_maps()

    def _save_video(self, name):
        """Save images or videos of the visualizations if offline mode is
        enabled."""

        print(f"Saving {name}.png")
        self.vm.fig.savefig(f"{name}.png", dpi=300)
        print(f"Saving {name}.gif")
        self.vm.mk_video(name=name, dirname=".")

    def _create_palette(self, T=False):
        """
        Create a 2D palette to visualize the weights.

        Parameters:
        - T (bool): Transpose flag for the palette. Defaults to False.

        Returns:
        - Numpy array representing the color palette.
        """
        palette1 = plt.cm.jet(np.linspace(0.1, 0.9, self.side))
        palette2 = plt.cm.CMRmap(np.linspace(0.1, 0.9, self.side))
        combined_palette = create_2d_palette(palette1, palette2)
        if T:
            return combined_palette.transpose(1, 0, 2).reshape(-1, 4)
        return combined_palette.reshape(-1, 4)

    def _prepare_grid(self):
        """Prepare a grid for scatter plots."""
        t = np.linspace(0, self.fovea_size * (self.side - 1), self.side)
        t += 0.2 * self.fovea_size
        return np.stack([x.ravel() for x in np.meshgrid(t, t[::-1])])

    def _initialize_maps(self):
        """Initialize all map visualizations."""
        self._initialize_visual_conditions_map()
        self._initialize_visual_effects_map()
        self._initialize_attention_map_traces()
        self._set_axis_limits()

    def _initialize_visual_conditions_map(self):
        """Initialize the visual conditions map image."""
        initial_shape = self.reshape_visual_weights(
            self.controller.visual_conditions_map.weights
        ).shape
        self.visual_conditions_map_im = self.visual_conditions_map_ax.imshow(
            np.zeros(initial_shape), vmin=0, vmax=1, zorder=0
        )
        self.visual_conditions_map_ax.set_xlim(0, self.side * self.fovea_size)
        self.visual_conditions_map_ax.set_ylim(0, self.side * self.fovea_size)
        self.visual_conditions_map_states = (
            self.visual_conditions_map_ax.scatter(
                *self.grid, s=30, fc=self.palette, ec="#000", zorder=1
            )
        )
        self.visual_conditions_map_focus = self._create_focus_scatter(
            self.visual_conditions_map_ax
        )

    def _create_focus_scatter(self, ax):
        """Create a scatter plot to indicate saccade."""
        return ax.scatter(
            1e100, 1e100, linewidth=6, s=120, fc="#fff0", ec="#a00f", zorder=1
        )

    def _initialize_visual_effects_map(self):
        """Initialize the visual effects map image."""
        initial_shape = self.reshape_visual_weights(
            self.controller.visual_effects_map.weights
        ).shape
        self.visual_effects_map_im = self.visual_effects_map_ax.imshow(
            np.zeros(initial_shape), vmin=0, vmax=1, zorder=0
        )
        self.visual_effects_map_ax.set_xlim(0, self.side * self.fovea_size)
        self.visual_effects_map_ax.set_ylim(0, self.side * self.fovea_size)
        self.visual_effects_map_states = self.visual_effects_map_ax.scatter(
            *self.grid, s=30, fc=self.palette, ec="#000", zorder=1
        )
        self.visual_effects_map_focus = self._create_focus_scatter(
            self.visual_effects_map_ax
        )

    def _initialize_attention_map_traces(self):
        """Initialize traces for the attention map."""
        initial_shape = self.controller.attention_map.weights.shape
        self.attention_map_im = self.attention_map_ax.scatter(
            *np.zeros(initial_shape), c=self.att_palette, s=120, zorder=1
        )
        self.attention_map_focus = self.attention_map_ax.scatter(
            1e100, 1e100, s=40, ec="black", fc="#ffff", lw=3
        )

        num_traces = 2 * self.side
        self.attention_map_traces = [
            self.attention_map_ax.plot(0, 0, color="black", zorder=0)[0]
            for _ in range(num_traces)
        ]

    def _set_axis_limits(self):
        """Set limits for the map axes."""
        self.attention_map_ax.set_xlim(
            [-0.1 * self.env_height, 1.1 * self.env_height]
        )
        self.attention_map_ax.set_ylim(
            [-0.1 * self.env_width, 1.1 * self.env_width]
        )

    def step(self, saccade=None):
        """
        Updates the displayed fovea map with the latest weights from the
        controller.

        Parameters:
        - saccade (np.array, optional): saccade point in the fovea weights,
          defaults to None.

        """
        if saccade is not None:
            self.saccade = saccade.ravel().astype(int)
        self._update_maps()
        self.fig.canvas.draw_idle()
        self.vm.save_frame()

    def _update_maps(self):
        """Update all component maps with new weights."""
        self._update_visual_conditions_map()
        self._update_visual_effects_map()
        self._update_attention_map_weights()

    def _update_visual_conditions_map(self):
        """Update visual conditions map with new weights."""
        weights = self.reshape_visual_weights(
            self.controller.visual_conditions_map.weights
        )

        if np.min(weights) != np.max(weights):
            self.visual_conditions_map_im.set_data(
                self._normalize_weights(weights)
            )

        self._update_focus(self.visual_conditions_map_focus)

    def _update_visual_effects_map(self):
        """Update visual effects map with new weights."""
        weights = self.reshape_visual_weights(
            self.controller.visual_effects_map.weights
        )

        if np.min(weights) != np.max(weights):
            self.visual_effects_map_im.set_data(
                self._normalize_weights(weights)
            )

        self._update_focus(self.visual_effects_map_focus)

    def _update_focus(self, focus_pointer):
        """Update the saccade pointer position based on current saccade data."""
        saccade = (self.saccade or 1e100 * np.ones(2)).ravel()
        saccade = self.fovea_size * (0.2 + np.array([saccade[1], saccade[0]]))
        saccade_pointer.set_offsets(saccade)

    def _update_attention_map_weights(self):
        """Update attention map weights."""
        weights = (
            self.controller.attention_map.weights.clone()
            .cpu()
            .detach()
            .numpy()
            * np.array([self.env_height, self.env_width]).reshape(-1, 1)
        )
        weights = weights[::-1]

        self.attention_map_im.set_offsets(weights.T)

        num_traces = self.side
        reshaped_weights = weights.reshape(
            2, num_traces, num_traces
        ).transpose(2, 1, 0)[::-1, :, :]
        for p in range(num_traces):
            self.attention_map_traces[p].set_data(*reshaped_weights[p, :, :].T)
        for p in range(num_traces, 2 * num_traces):
            self.attention_map_traces[p].set_data(
                *reshaped_weights[:, p % num_traces, :].T
            )

        if self.saccade is not None:
            try:
                self.attention_map_focus.set_offsets(
                    reshaped_weights[self.saccade[0], self.saccade[1], :]
                )
            except IndexError:
                print("Index error")
                self.attention_map_saccade.set_offsets(self.saccade)

    def _normalize_weights(self, weights):
        """
        Normalize the given weights to the range [0, 1].

        Parameters:
        - weights (np.array): The weights to normalize.

        Returns:
        - np.array: The normalized weights.
        """
        return (weights - np.min(weights)) / (np.ptp(weights))

    def close(self, name=None):
        """Close any resources and optionally save the maps to a file."""
        if self.offline and name:
            self._save_video(name)
        plt.close(self.fig)

    def reshape_visual_weights(self, weights):
        """
        Reshape and transpose the fovea map weights for plotting.

        Parameters:
        - weights (np.array): The weights to be reshaped.

        Returns:
        - np.array: Reshaped and transposed weights for visualization.
        """
        inp_side1, inp_side2 = self.env.fovea_size.astype(int)
        out_side1 = out_side2 = self.side

        reshaped_weights = (
            weights.cpu()
            .detach()
            .numpy()
            .reshape(inp_side1, inp_side2, 3, out_side1, out_side2)[::-1]
        )
        transposed_weights = reshaped_weights.transpose(3, 0, 4, 1, 2).reshape(
            inp_side1 * out_side1, inp_side2 * out_side2, 3
        )

        return transposed_weights


def create_2d_palette(palette1, palette2):
    """
    Create a 2D color palette from two 1D color palettes.

    Parameters:
    - palette1: List of colors (hex strings, RGB tuples, etc.)
    - palette2: List of colors (hex strings, RGB tuples, etc.)

    Returns:
    - 2D array of combined colors
    """
    N, M = len(palette1), len(palette2)
    combined_palette = np.zeros((N, M, 4))
    for i in range(N):
        for j in range(M):
            combined_palette[i, j] = (palette1[i] + palette2[j]) / 2
            combined_palette[i, j][3] = 1  # Set alpha to fully opaque

    return combined_palette
