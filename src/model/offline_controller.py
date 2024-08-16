import numpy as np
import torch
from model.topological_maps import (
    TopologicalMap,
    STMUpdater,
)


class OfflineController:
    def __init__(self, env):

        input_batch_size = 1000
        maps_output_size = 10 * 10

        self.base_learning_rate = 0.001
        self.learning_rate_range = 0.1
        self.learning_rate = 0
        self.base_std = np.sqrt(2)
        self.std_range = 8.0
        self.std = 0

        self.env = env

        self.set_hyperparams(competence=0.0)

        # Get observation space sample shape and calculate the input size
        fovea_sample = self.env.observation_space['FOVEA'].sample()
        fovea_input_size = fovea_sample.size

        # Initialize TopologicalMap and STMUpdater
        self.fovea_map = TopologicalMap(fovea_input_size, maps_output_size)
        self.fovea_updater = STMUpdater(self.fovea_map, self.base_learning_rate)

        # Preallocate tensors for batch processing
        fovea_batch_shape = (
            input_batch_size,
            maps_output_size,
        )
        self.fovea_inputs = torch.zeros((input_batch_size, fovea_input_size))
        self.fovea_grid_representations = torch.zeros(fovea_batch_shape)
        self.fovea_point_representations = torch.zeros((input_batch_size, 2))

        # Initialize counter for fovea inputs
        self.fovea_input_counter = 0

    def set_hyperparams(self, competence):
        exp_factor = np.exp(-(1 - competence))

        self.learning_rate = (
            self.base_learning_rate + self.learning_rate_range * exp_factor
        )
        self.std = self.base_std + self.std_range * exp_factor

    def set_goal(self):
        pass

    def store_fovea_input(self):

        # Flatten and reshape the fovea input
        fovea_input = self.env.observation['FOVEA'].ravel()
        fovea_reshaped = torch.tensor(fovea_input).unsqueeze(0)

        # Obtain fovea output
        fovea_output = self.fovea_map(fovea_reshaped, self.std)

        # Update fovea inputs and representations
        index = self.fovea_input_counter
        self.fovea_inputs[index] += fovea_input
        self.fovea_grid_representations[
            index
        ] += self.fovea_map.get_representation('grid').ravel()
        self.fovea_point_representations[
            index
        ] += self.fovea_map.get_representation('point').ravel()

        self.fovea_input_counter += 1

    def update_maps(self):

        # Slicing the tensor to only include relevant fovea outputs
        inputs = self.fovea_inputs[: self.fovea_input_counter]
        fovea_outputs = self.fovea_map(inputs, self.std)

        # Updating the fovea using the extracted outputs
        self.fovea_updater(
            fovea_outputs,
            torch.ones_like(fovea_outputs),
            self.learning_rate,
        )

        # Resetting tensors to zero using in-place operations to maintain memory efficiency
        for tensor in [
            self.fovea_inputs,
            self.fovea_grid_representations,
            self.fovea_point_representations,
        ]:
            tensor.zero_()
