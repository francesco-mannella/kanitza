#%% IMPORTS
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import EyeSim

class FoveaPlotter(EyeSim.envs.Simulator.TestPlotter):
    """
    FoveaPlotter is a custom plotter for visualizing the saliency map, 
    fovea region, and salient points in an eye simulation environment.
    """

    def __init__(self, env):
        """
        Initializes the FoveaPlotter with a given environment.

        Parameters:
        env (object): The simulation environment containing retina and fovea configurations.
        """
        self.env = env
        
        # Create the figure and axes
        self.fig, self.axes = plt.subplots(1, 4, figsize=(10, 3), 
                                           gridspec_kw={'width_ratios': [1, 1, 1, 1], 
                                                        'height_ratios': [1]})
        
        # Initialize the saliency image plot and highlight dot
        self.saliency_image = self.axes[0].imshow(np.zeros(env.retina_size), vmin=0, vmax=1)
        self.attentional_mask = self.axes[3].imshow(np.zeros(env.retina_size), vmin=0, vmax=1)
        self.highlight_dot = self.axes[0].scatter(0, 0, c='#ff5555', s=50, alpha=0.5 )
        
        # Initialize the fovea image plot
        self.fovea_image = self.axes[2].imshow(env.observation_space["FOVEA"].sample(), vmin=0, vmax=1)
        
        # Create the rectangles for retina and fovea positions
        self._initialize_patches()

        # Initialize TestPlotter
        super().__init__(env, ax=self.axes[1])

        # Set aspect ratio for all axes
        for ax in self.axes:
            ax.set_box_aspect(1)
    
    def _initialize_patches(self):
        """Helper method to initialize the retina and fovea position rectangles."""
        pos = self.env.retina_sim_pos - self.env.retina_scale / 2
        self.retina_pos = Rectangle(pos, *self.env.retina_scale, lw=0.5, fill=None, color="#888")
        
        pos = self.env.retina_sim_pos - self.env.fovea_scale / 2
        self.fovea_pos = Rectangle(pos, *self.env.fovea_scale, fill=None)
        
        self.axes[1].add_patch(self.retina_pos)
        self.axes[1].add_patch(self.fovea_pos)

    def render(self, saliency_map, salient_point, attentional_mask=None):
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
        self.fovea_image.set_array(self.env.observation["FOVEA"])

        # Update positions of rectangles
        self.retina_pos.set_xy(self.env.retina_sim_pos - self.env.retina_scale / 2)
        self.fovea_pos.set_xy(self.env.retina_sim_pos - self.env.fovea_scale / 2)

        # Ensure rectangles are drawn
        self.axes[1].add_patch(self.retina_pos)
        self.axes[1].add_patch(self.fovea_pos)

        # Pause to update the plot
        plt.pause(0.1)
