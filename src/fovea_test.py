import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.topological_maps import TopologicalMap, som_loss
import numpy as np


def train_som(som_model, loader, num_epochs):
    """Train a self-organizing map.

    Args:
        som_model (TopologicalMap): The SOM model to be trained.
        loader (object): The data loader object used to feed data to the model.
        num_epochs (int): The number of epochs to train the model for.

    Returns:
        learning_rates (list): List of learning rate values for each epoch.
        epoch_losses (list): List of loss values for each epoch.
        activations_per_epoch (list): List of activation data for each epoch.
        weights_per_epoch (list): List of weight data for each epoch.
    """
    
    # Initialize hyperparameters
    initial_learning_rate = 0.1
    final_learning_rate = 1e-7
    learning_rate_decay = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    final_std_deviation = 1e-6
    std_deviation_decay = final_std_deviation ** (1 / num_epochs) 
    initial_std_deviation = 1
    
    # Initialize optimizer for model parameters
    optimizer = torch.optim.Adam(som_model.parameters(), lr=initial_learning_rate)
    
    # Initialize lists to store output values
    learning_rates = []
    epoch_losses = []
    activations_per_epoch = []
    weights_per_epoch = []

    # Iterate over epochs
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Calculate standard deviation for current epoch
        current_std_deviation = initial_std_deviation * std_deviation_decay ** epoch
        
        # Calculate learning rate for current epoch
        current_learning_rate = initial_learning_rate * learning_rate_decay ** epoch

        # Iterate over data batches
        for batch_data in loader:
            inputs = batch_data

            optimizer.zero_grad()
            
            # Forward pass through the model
            outputs = som_model(inputs, current_std_deviation)
            
            # Calculate loss
            som_loss_value = som_loss(outputs)
            loss = current_learning_rate * som_loss_value
            
            # Backward pass and update gradients
            loss.backward()
            optimizer.step()

            total_loss += som_loss_value.item()
        
        # Log end of epoch loss
        print(f'Epoch [{epoch + 1}/{num_epochs}] loss: {total_loss:.5f}')
        
        # Append values to corresponding lists
        learning_rates.append(current_learning_rate)
        epoch_losses.append(total_loss)
        activations_per_epoch.append(np.stack(som_model.get_representation("grid")))
        weights_per_epoch.append(np.stack(som_model.weights.tolist()))
    
    # Return output values
    return learning_rates, epoch_losses, activations_per_epoch, weights_per_epoch

class NumpyDataset(Dataset):
    def __init__(self, numpy_data):
        self.data = numpy_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float)

# load dataset
input_files = ['inputs_000.npy', 'inputs_001.npy']
inputs = np.vstack([np.load(f) for f in input_files])

# Determine the shape of the inputs
input_num, input_size = inputs.shape

# Define the output size
output_size = 100

# Create dataset and dataloader
numpy_dataset = NumpyDataset(inputs)
data_loader = DataLoader(numpy_dataset, batch_size=200, shuffle=True, num_workers=2)


# create topological map with 100 units
topological_map = TopologicalMap(input_size, output_size)
train_som(topological_map, data_loader, num_epochs=30) 
        
def reshape_weights(weights):
    inp_side1, inp_side2 = 16, 16
    out_side1, out_side2 = 10, 10

    # Convert weights to numpy array and reshape
    weights = weights.cpu().detach().numpy()
    reshaped_weights = weights.reshape(
        inp_side1, inp_side2, 3, out_side1, out_side2
    )

    # Transpose and reshape the numpy array
    transposed_weights = reshaped_weights.transpose(3, 0, 4, 1, 2)
    new_shape = (inp_side1 * out_side1, inp_side2 * out_side2, 3)
    reshaped_transposed_weights = transposed_weights.reshape(new_shape)

    return reshaped_transposed_weights

import matplotlib.pyplot as plt

# Reshape the topological_map weights
reshaped_weights = reshape_weights(topological_map.weights)

# Plot the reshaped weights using imshow
plt.imshow(reshaped_weights)
plt.title("Topological Map Weights")
plt.axis('off')  # Hide the axis
plt.show()
