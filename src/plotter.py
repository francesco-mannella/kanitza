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
            np.zeros_like(env.observation_space['FOVEA'].sample()), vmin=0, vmax=1
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
            1, 2, figsize=(6, 3),
            gridspec_kw={'width_ratios': np.ones(2), 'height_ratios': [1]},
        )

        # Initialize imshow objects for both axes
        fovea_map_weights = self.reshape_fovea_weights(
            self.controller.fovea_map.weights,
        )
        self.fovea_map_im = self.fovea_map_ax.imshow(
            fovea_map_weights,
        )
    
        # Initialize the attentional map with a red scatter plot at the origin
        self.attentional_map_im = self.attentional_map_ax.scatter(0, 0, color="red", s=20)

        # Determine the number of traces to plot
        num_traces = 2 * int(np.sqrt(self.controller.maps_output_size))

        # Create a list of black line plots for the attentional map traces
        self.attentional_map_traces = [
            self.attentional_map_ax.plot(0, 0, color="black")[0]
            for _ in range(num_traces)
        ]

        # Set the x and y limits of the attentional map axis based on retina scale
        x_lim = 0.7 * self.env.retina_scale[0]
        y_lim = 0.7 * self.env.retina_scale[1]
        self.attentional_map_ax.set_xlim([-x_lim, x_lim])
        self.attentional_map_ax.set_ylim([-y_lim, y_lim])

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

        # Get the attentional map weights from the controller and convert them to numpy array
        attentional_map_weights = self.controller.attentional_map.weights.clone().cpu().detach().numpy()

        # Adjust the weights using the retina scale from the environment
        retina_scale_reshaped = self.env.retina_scale.reshape(-1, 1)
        attentional_map_weights *= retina_scale_reshaped
        attentional_map_weights -= retina_scale_reshaped / 2

        # Update the offsets of the attentional map image for display
        self.attentional_map_im.set_offsets(attentional_map_weights.T)

        # Determine the number of traces to plot
        num_traces = int(np.sqrt(self.controller.maps_output_size))

        # Reshape and transpose the weights for simplified plotting
        attentional_map_weights = attentional_map_weights.reshape(2, num_traces, num_traces).transpose(1, 2, 0)

        # Update the positional data of each trace plot
        for p in range(num_traces):
            curr_plot = self.attentional_map_traces[p]
            curr_plot.set_data(*attentional_map_weights[p, :, :].T)

        for p in range(num_traces, 2 * num_traces):
            curr_plot = self.attentional_map_traces[p]
            curr_plot.set_data(*attentional_map_weights[:, p % num_traces, :].T)

        # Redraw the figure to show updates
        self.fig.canvas.draw_idle()


    def close(self, name=None):
        if self.offline and name is not None:
            print(f"save {name}")
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
