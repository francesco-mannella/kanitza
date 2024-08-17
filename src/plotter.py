#%% IMPORTS
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

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
            figsize=(10, 3),
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
            env.observation_space['FOVEA'].sample(), vmin=0, vmax=1
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


class MapsPlotter:
    """
    A class for plotting fovea and attentional maps using matplotlib.

    This class initializes with a controller that provides the necessary map data
    and sets up the figure and axis for plotting.
    """

    def __init__(self, env, controller, offline=False):
        """
        Initializes the MapsPlotter with the given controller and environment.

        Args:
            env (object): The simulation environment containing retina and fovea configurations.
            controller (object): An object that contains fovea_map and attentional_map, each with weights.
        """
        self.controller = controller
        self.env = env

        # Setup the figure and axis for plotting.
        self.fig, (self.fovea_map_ax, self.attentional_map_ax) = plt.subplots(
            1, 2, figsize=(10, 3)
        )

        # Initialize imshow objects for both axes
        fovea_map_weights = self.reshape_fovea_weights(
            self.controller.fovea_map.weights,
        )
        self.fovea_map_im = self.fovea_map_ax.imshow(
            fovea_map_weights, cmap='viridis'
        )
        # self.attentional_map_im = self.attentional_map_ax.imshow(
        #     self.controller.attentional_map.weights, cmap='viridis'
        # )

        self.offline = offline

    def step(self):
        """
        Updates the displayed fovea map with the latest weights from the controller.

        This method retrieves the current fovea map weights from the controller, reshapes
        and transposes them as needed for visualization, and updates the imshow object.
        """

        # Reshape and transpose fovea weights for correct visualization
        fovea_map_weights = self.reshape_fovea_weights(
            self.controller.fovea_map.weights,
        )

        # Set the updated data on the imshow object
        self.fovea_map_im.set_data(fovea_map_weights)

        # Redraw the figure to show updates
        self.fig.canvas.draw_idle()


    def close(self, name=None):
        if self.offline and name is not None:
            self.fig.savefig(name, dpi=300)
        plt.close(self.fig)

    def reshape_fovea_weights(
        self,
        weights,
    ):
        """
        Reshapes and transposes the fovea map weights for plotting.

        Args:
            weights (torch.Tensor): The fovea map weights to be reshaped.

        Returns:
            np.ndarray: Reshaped and transposed weights ready for visualization.
        """
        inp_side1, inp_side2 = self.env.fovea_size.astype(int)
        out_side1, out_side2 = np.ones(2).astype(int) * int(
            np.sqrt(self.controller.maps_output_size)
        )

        weights = weights.cpu().detach().numpy()
        np.save("weights", weights)
        reshaped_weights = weights.reshape(
            inp_side1, inp_side2, 3, out_side1, out_side2,
        )
        transposed_weights = reshaped_weights.transpose(3, 0, 4, 1, 2)
        new_shape = (inp_side1 * out_side1, inp_side2 * out_side2, 3)

        reshaped_transposed_weights = transposed_weights.reshape(new_shape)

        return reshaped_transposed_weights
