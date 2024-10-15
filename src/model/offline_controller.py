import numpy as np
import torch
from model.topological_maps import (
    TopologicalMap,
    STMUpdater,
)


class OfflineController:
    """
    OfflineController class manages the update of topological maps using
    sensory inputs from an environment.

    """

    def __init__(
        self,
        env,
        params,
        seed = None,
    ):
        """
        Initialize the OfflineController with the given environment.

        """

        # Save the environment reference
        self.env = env

        # save parameters reference
        self.params = params

        if seed is None:
            seed = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)


    def set_hyperparams(self):
        pass
    
    def generate_attentional_input(self, focus_num):
        
        return np.array([np.random.choice(np.arange(size), focus_num) for size in self.env.retina_size]).T

    def update_maps(self):
        pass
    
    def update_predicts(self):
        pass


