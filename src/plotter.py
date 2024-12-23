#%% IMPORTS
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from params import Parameters


import EyeSim
from EyeSim.envs.mkvideo import vidManager


class FoveaPlotter(EyeSim.envs.Simulator.TestPlotter):
    """
    FoveaPlotter is a custom plotter for visualizing the saliency map,
    fovea region, and salient points in an eye simulation environment.
    """

    def __init__(self, env, *args, **kargs):
        """
        Initializes the FoveaPlotter with a given environment.

        Parameters:
        env (object): The simulation environment containing retina and fovea configurations.
        """
        self.env = env

        # Create the figure and axes
        self.fig, self.axes = plt.subplots(
            1,
            4,
            figsize=(9, 3),
            gridspec_kw={'width_ratios': np.ones(4), 'height_ratios': [1]},
        )

        (
            self.env_ax,
            self.fovea_ax,
            self.saliency_ax,
            self.filter_ax,
        ) = self.axes

        # Initialize the saliency image plot and highlight dot
        self.saliency_image = self.saliency_ax.imshow(
            np.zeros(env.retina_size), vmin=0, vmax=1
        )
        self.attentional_mask = self.filter_ax.imshow(
            np.zeros(env.retina_size), vmin=0, vmax=1
        )
        self.highlight_dot = self.saliency_ax.scatter(
            0, 0, c='#ff5555', s=50, alpha=0.5
        )

        # Initialize the fovea image plot
        self.fovea_image = self.fovea_ax.imshow(
            np.zeros_like(env.observation_space['FOVEA'].sample()),
            vmin=0,
            vmax=1,
        )

        # Create the rectangles for retina and fovea positions
        self._initialize_patches()

        # Initialize TestPlotter
        super().__init__(env, ax=self.env_ax, *args, **kargs)

        # Set aspect ratio for all axes
        for ax in self.axes:
            ax.set_box_aspect(1)

        self.env_ax.set_title('Visual field')
        self.fovea_ax.set_title('Fovea')
        self.saliency_ax.set_title('Retina\n(salience)')
        self.filter_ax.set_title('Retina\n(attentional filter)')

    def _initialize_patches(self):
        """Helper method to initialize the retina and fovea position rectangles."""
        pos = self.env.retina_sim_pos - self.env.retina_scale / 2
        self.retina_pos = Rectangle(
            pos, *self.env.retina_scale, lw=0.5, fill=None, color='#888'
        )

        pos = self.env.retina_sim_pos - self.env.fovea_scale / 2
        self.fovea_pos = Rectangle(pos, *self.env.fovea_scale, fill=None)

        self.env_ax.add_patch(self.retina_pos)
        self.env_ax.add_patch(self.fovea_pos)

    def step(self, saliency_map, salient_point, attentional_mask=None):
        """
        Renders the current state of the environment, updating the saliency map,
        fovea image, and highlights salient points.

        Parameters:
        saliency_map (np.array): The current saliency map to display.
        salient_point (tuple): Coordinates of the salient point to highlight.
        attentiolnal_mask: the current attentional mask of salience
        """
        super().step()

        # Update the saliency image and the highlight dot
        self.saliency_image.set_array(saliency_map)
        if attentional_mask is not None:
            self.attentional_mask.set_array(attentional_mask)
        self.highlight_dot.set_offsets(salient_point)

        # Update the fovea image
        self.fovea_image.set_array(self.env.observation['FOVEA'])

        # Update positions of rectangles
        self.retina_pos.set_xy(
            self.env.retina_sim_pos - self.env.retina_scale / 2
        )
        self.fovea_pos.set_xy(
            self.env.retina_sim_pos - self.env.fovea_scale / 2
        )

        # Ensure rectangles are drawn
        self.env_ax.add_patch(self.retina_pos)
        self.env_ax.add_patch(self.fovea_pos)

        # Redraw the figure to show updates
        self.fig.canvas.draw_idle()


class FakeMapsPlotter:
    """
    A class for a fake mapS_plotter which does not do anything.
    """

    def __init__(self, env, controller, offline=False):
        """
        Initializes the MapsPlotter with a fake implementation that doesn't perform any plotting.
        """
        pass

    def step(self):
        """
        A method that doesn't do anything. It's just a placeholder.
        """
        pass

    def close(self, name=None):
        """
        A method that doesn't save any figures or close anything, just a dummy function.
        """
        pass

    def reshape_fovea_weights(self, weights):
        """
        A method that doesn't perform any reshaping of weights and returns None.
        """
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
        """
        self.controller = controller
        self.env = env
        self.env_height, self.env_width = env.observation_space[
            'RETINA'
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
        ) = self._create_figure()
        self.vm = vidManager(
            self.fig, name='maps', dirname='.', duration=video_frame_duration
        )

        self.grid = self._prepare_grid()

        self.focus = None

        self._initialize_maps()

    def _create_palette(self, T=False):
        palette1 = plt.cm.jet(np.linspace(0.1, 0.9, self.side))
        palette2 = plt.cm.CMRmap(np.linspace(0.1, 0.9, self.side))
        if T == True:
            return create_2d_palette(palette1, palette2)[::-1, :, :].reshape(
                -1, 4
            )

        return create_2d_palette(palette1, palette2).reshape(-1, 4)

    def _create_figure(self):
        return plt.subplots(1, 3, figsize=(9, 3))

    def _prepare_grid(self):
        t = np.linspace(0, self.fovea_size * (self.side - 1), self.side)
        t += 0.2 * self.fovea_size
        return np.stack([x.ravel() for x in np.meshgrid(t, t[::-1])])

    def _initialize_maps(self):
        self._initialize_visual_conditions_map()
        self._initialize_visual_effects_map()
        self._initialize_attention_map_traces()
        self._set_axis_limits()

    def _initialize_visual_conditions_map(self):
        initial_shape = self.reshape_visual_weights(
            self.controller.visual_conditions_map.weights
        ).shape
        self.visual_conditions_map_im = self.visual_conditions_map_ax.imshow(
            np.zeros(initial_shape), vmin=0, vmax=1, zorder=0
        )
        side = self.fovea_size * self.side
        self.visual_conditions_map_ax.set_xlim(0, side)
        self.visual_conditions_map_ax.set_ylim(0, side)
        self.visual_conditions_map_states = (
            self.visual_conditions_map_ax.scatter(
                *self.grid, s=30, fc=self.palette, ec='#000', zorder=1
            )
        )
        self.visual_conditions_map_focus = (
            self.visual_conditions_map_ax.scatter(
                1e100,
                1e100,  # Start point, off the visible figure initially
                linewidth=6,
                s=120,
                fc='#fff0',
                ec='#a00f',
                zorder=1,
            )
        )

    def _initialize_visual_effects_map(self):
        initial_shape = self.reshape_visual_weights(
            self.controller.visual_effects_map.weights
        ).shape
        self.visual_effects_map_im = self.visual_effects_map_ax.imshow(
            np.zeros(initial_shape), vmin=0, vmax=1, zorder=0
        )
        side = self.fovea_size * self.side
        self.visual_effects_map_ax.set_xlim(0, side)
        self.visual_effects_map_ax.set_ylim(0, side)
        self.visual_effects_map_states = self.visual_effects_map_ax.scatter(
            *self.grid, s=30, fc=self.palette, ec='#000', zorder=1
        )
        self.visual_effects_map_focus = self.visual_effects_map_ax.scatter(
            1e100,
            1e100,  # Start point, off the visible figure initially
            linewidth=6,
            s=120,
            fc='#fff0',
            ec='#a00f',
            zorder=1,
        )

    def _initialize_attention_map_traces(self):
        initial_shape = self.controller.attention_map.weights.shape
        self.attention_map_im = self.attention_map_ax.scatter(
            *np.zeros(initial_shape), c=self.att_palette, s=120, zorder=1
        )

        self.attention_map_focus = self.attention_map_ax.scatter(
            1e100, 1e100, s=40, ec='black', fc='#ffff', lw=3
        )
        num_traces = 2 * self.side
        self.attention_map_traces = [
            self.attention_map_ax.plot(0, 0, color='black', zorder=0)[0]
            for _ in range(num_traces)
        ]

    def _set_axis_limits(self):
        y = self.env_height
        x = self.env_width
        self.attention_map_ax.set_xlim([-0.1 * y, 1.1 * y])
        self.attention_map_ax.set_ylim([1.1 * x, -0.1 * x])

    def step(self, focus=None):
        """
        Updates the displayed fovea map with the latest weights from the controller.
        """

        if focus is not None:
            self.focus = focus.ravel().astype(int)

        self._update_maps()
        self.fig.canvas.draw_idle()
        self.vm.save_frame()

    def _update_maps(self):
        self._update_visual_conditions_map()
        self._update_visual_effects_map()
        self._update_attention_map_weights()

    def _update_visual_conditions_map(self):
        weights = self.reshape_visual_weights(
            self.controller.visual_conditions_map.weights
        )
        if np.min(weights) != np.max(weights):
            self.visual_conditions_map_im.set_data(
                self._normalize_weights(weights)
            )
        if self.focus is not None:
            focus = self.focus.ravel()
        else:
            focus = 1e100 * np.ones(2)

        focus = self.fovea_size * (0.2 + (np.array([focus[1], focus[0]])))
        self.visual_conditions_map_focus.set_offsets(
            focus
        )  # adjusting to position center

    def _update_visual_effects_map(self):
        weights = self.reshape_visual_weights(
            self.controller.visual_effects_map.weights
        )
        if np.min(weights) != np.max(weights):
            self.visual_effects_map_im.set_data(
                self._normalize_weights(weights)
            )
        if self.focus is not None:
            focus = self.focus.ravel()
        else:
            focus = 1e100 * np.ones(2)

        focus = self.fovea_size * (0.2 + (np.array([focus[1], focus[0]])))
        self.visual_effects_map_focus.set_offsets(
            focus
        )  # adjusting to position center

    def _update_attention_map_weights(self):
        weights = (
            self.controller.attention_map.weights.clone()
            .cpu()
            .detach()
            .numpy()
        )
        retina_scale_reshaped = self.env.retina_scale.reshape(-1, 1)

        env_size = np.array([self.env_height, self.env_width])
        weights *= env_size.reshape(-1, 1)
        weights = weights[::-1]

        self.attention_map_im.set_offsets(weights.T)

        num_traces = self.side
        reshaped_weights = weights.reshape(
            2, num_traces, num_traces
        ).transpose(1, 2, 0)
        for p in range(num_traces):
            self.attention_map_traces[p].set_data(*reshaped_weights[p, :, :].T)
        for p in range(num_traces, 2 * num_traces):
            self.attention_map_traces[p].set_data(
                *reshaped_weights[:, p % num_traces, :].T
            )

        if self.focus is not None:
            try:
                self.attention_map_focus.set_offsets(
                    reshaped_weights[self.focus[0], self.focus[1], :]
                )
            except IndexError:
                print('index error')
                self.attention_map_focus.set_offsets(self.focus)

    def _normalize_weights(self, weights):
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        return (weights - min_weight) / (max_weight - min_weight)

    def close(self, name=None):
        if self.offline and name is not None:
            print(f'save {name}.png')
            self.vm.fig.savefig(f'{name}.png', dpi=300)
            print(f'save {name}.gif')
            self.vm.mk_video(name=name)
        plt.close(self.fig)

    def reshape_visual_weights(self, weights):
        """
        Reshapes and transposes the fovea map weights for plotting.
        """
        inp_side1, inp_side2 = self.env.fovea_size.astype(int)
        out_side1 = out_side2 = self.side

        weights = weights.cpu().detach().numpy()
        reshaped_weights = weights.reshape(
            inp_side1, inp_side2, 3, out_side1, out_side2
        )
        reshaped_weights = reshaped_weights[::-1, :, :, :, :]
        transposed_weights = reshaped_weights.transpose(3, 0, 4, 1, 2)

        transposed_weights = transposed_weights.reshape(
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

    N = len(palette1)
    M = len(palette2)
    # Create a 2D array of colors by combining the two palettes
    combined_palette = np.zeros((N, M, 4))
    for i in range(N):
        for j in range(M):
            # Mix shades from both palettes based on their indices
            combined_palette[i, j] = (
                palette1[i] + palette2[j]
            ) / 2  # Average RGB values
            combined_palette[i, j][
                3
            ] = 1  # Set alpha channel to 1 (fully opaque)

    return combined_palette
