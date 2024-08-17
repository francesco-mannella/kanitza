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

    Attributes:
        base_learning_rate (float): Base learning rate for updates.
        learning_rate_range (float): Range variation for the adaptive learning rate.
        learning_rate (float): Current adaptive learning rate.
        base_std (float): Base standard deviation for sensory input noise.
        std_range (float): Range variation for the adaptive standard deviation.
        std (float): Current adaptive standard deviation.
        env: Environment providing sensory inputs.
        fovea_map: Topological map for fovea sensory input.
        fovea_updater: STMUpdater instance for updating the fovea map.
        fovea_inputs (torch.Tensor): Preallocated tensor for batch of fovea inputs.
        fovea_grid_representations (torch.Tensor): Preallocated tensor for fovea grid representations.
        fovea_point_representations (torch.Tensor): Preallocated tensor for fovea point representations.
        fovea_input_counter (int): Counter for tracking the number of stored fovea inputs.
    """

    def __init__(self, env):
        """
        Initialize the OfflineController with the given environment.

        Args:
            env: An environment object which has an observation_space
                 dictionary with a 'FOVEA' key.
        """

        # Initialize input batch size and map output size
        self.input_batch_size = 20000  # Batch size for input processing
        self.maps_output_size = (
            10 * 10
        )  # Size of the output map representation
        self.batch_shape = (self.input_batch_size, self.maps_output_size)

        # Initialize learning rate and standard deviation parameters
        self.base_learning_rate = 0.001
        self.learning_rate_range = 1
        self.learning_rate = 1
        self.base_std = 0.5 * np.sqrt(2)
        self.std_range = 2.0
        self.std = 1

        # Save the environment reference
        self.env = env

        # Get observation space sample shape and calculate the input size
        fovea_sample = self.env.observation_space['FOVEA'].sample()
        self.fovea_input_size = fovea_sample.size

        # Initialize TopologicalMap and STMUpdater
        self.fovea_map = TopologicalMap(
            self.fovea_input_size, self.maps_output_size
        )
        self.fovea_updater = STMUpdater(self.fovea_map, 0.1)

        # Preallocate tensors for batch processing
        self.fovea_inputs = torch.zeros(
            (self.input_batch_size, self.fovea_input_size)
        )
        self.fovea_grid_representations = torch.zeros(self.batch_shape)
        self.fovea_point_representations = torch.zeros(
            (self.input_batch_size, 2)
        )

        # Initialize counter for fovea inputs
        self.fovea_input_counter = 0

        # attentional input
        self.attentional_input_size = 2
        
        # Initialize TopologicalMap and STMUpdater
        self.attentional_map = TopologicalMap(
            self.attentional_input_size, self.maps_output_size
        )
        self.attentional_updater = STMUpdater(self.attentional_map, 0.1)

        # Preallocate tensors for batch processing
        self.attentional_inputs = torch.zeros(
            (self.input_batch_size, self.attentional_input_size)
        )
        self.attentional_grid_representations = torch.zeros(self.batch_shape)
        self.attentional_point_representations = torch.zeros(
            (self.input_batch_size, 2)
        )

        # Initialize counter for attentional inputs
        self.attentional_input_counter = 0

    def set_hyperparams(self, competence):
        """
        Set the adaptive hyperparameters (learning rate and standard deviation)
        based on competence.

        Args:
            competence (float): A value indicating the current competence level.
        """
        exp_factor = 1 - competence

        self.learning_rate = (
            self.base_learning_rate + self.learning_rate_range * exp_factor
        )
        self.std = self.base_std + self.std_range * exp_factor

    def set_goal(self):
        """
        Placeholder for setting the goal in the environment.
        """
        pass

    def store_attentional_input(self, attentional_input):
        """
        Stores and processes attentional input.

        Args:
            attentional_input (np.ndarray): The input coordinates in the retina.

        TODO:
            The current encoding of attentional representations is FAKE
            and needs proper implementation.
        """

        # Store the attentional input at the current counter index
        index = self.attentional_input_counter
        self.attentional_inputs[index] += torch.tensor(attentional_input)

        # Rescale the attentional input to the ouput space
        # TODO this is a FAKE, the mapping should be learned
        scaled_attentional_input = self.rescale_attentional_input(
            attentional_input, self.env.retina_size, self.maps_output_size
        )


        # Create attentional grid representation using radial function
        self.attentional_grid_representations[
            index
        ] += self.attentional_map.radial(
            torch.tensor(scaled_attentional_input), 0.7, as_point=True
        ).squeeze()

        # Store the scaled attentional input as a point representation
        self.attentional_point_representations[
            index
        ] += scaled_attentional_input

        # Increment the input counter
        self.attentional_input_counter += 1

    def rescale_attentional_input(
        self, attentional_input, retina_size, maps_output_size
    ):
        """
        Rescales attentional input from original retina size to the map output size.

        Args:
            attentional_input (np.ndarray): The input coordinates in the retina.
            retina_size (tuple): The size of the retina.
            maps_output_size (int): The size of the map output.

        Returns:
            np.ndarray: The rescaled attentional input.
        """

        original_lowers = np.zeros(2)
        original_uppers = np.array(retina_size)

        new_lowers = np.zeros(2)
        new_uppers = np.ones(2) * np.sqrt(maps_output_size)

        # Perform rescaling operation
        scale = (new_uppers - new_lowers) / (original_uppers - original_lowers)
        scaled_attentional_input = (
            new_lowers + (attentional_input - original_lowers) * scale
        )

        scaled_attentional_input = (scaled_attentional_input - 4)*5
        return scaled_attentional_input

    def store_fovea_input(self):
        """
        Store a single fovea input to the preallocated tensors.
        """

        # Flatten and reshape the fovea input
        fovea_input = self.env.observation['FOVEA'].ravel()
        fovea_reshaped = torch.tensor(fovea_input).unsqueeze(0)

        # Obtain fovea output
        fovea_output = self.fovea_map(fovea_reshaped, self.std)

        # Update fovea inputs and representations
        index = self.fovea_input_counter
        self.fovea_inputs[index] += fovea_input / 255
        self.fovea_grid_representations[
            index
        ] += self.fovea_map.get_representation('grid').ravel()
        self.fovea_point_representations[
            index
        ] += self.fovea_map.get_representation('point').ravel()

        # Increment the input counter
        self.fovea_input_counter += 1

    def update_maps(self):
        """
        Update the topological maps using stored fovea inputs and reset the
        preallocated tensors.
        """
        # Slicing the tensor to only include relevant fovea outputs
        inputs = self.fovea_inputs[: self.fovea_input_counter]
        targets = self.attentional_grid_representations[
            : self.attentional_input_counter
        ]
        fovea_outputs = self.fovea_map(inputs, self.std)

        np.save('inputs', inputs.cpu().detach().numpy())
        # Updating the fovea map using the extracted outputs
        self.fovea_updater(fovea_outputs, targets, self.learning_rate)

        # Reset preallocated tensors to zero in-place for memory efficiency
        for tensor in [
            self.fovea_inputs,
            self.fovea_grid_representations,
            self.fovea_point_representations,
            self.attentional_inputs,
            self.attentional_grid_representations,
            self.attentional_point_representations,
        ]:
            tensor.zero_()
        self.fovea_input_counter = 0
        self.attentional_input_counter = 0
