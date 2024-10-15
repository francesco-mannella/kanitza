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
    ):
        """
        Initialize the OfflineController with the given environment.

        """

        # Save the environment reference
        self.env = env


    def set_hyperparams(self, competence):
        pass

    def update_maps(self):
        pass
    
    def update_predicts(self):
        pass


