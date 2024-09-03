import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.topological_maps import TopologicalMap, som_loss, stm_loss
import numpy as np
import matplotlib.pyplot as plt


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
    initial_learning_rate = 0.01
    final_learning_rate = 1e-9
    learning_rate_decay = (final_learning_rate / initial_learning_rate) ** (
        1 / (num_epochs*5)
    )
    initial_std_deviation = 3
    final_std_deviation = 0.5 * np.sqrt(2)
    std_deviation_decay = (final_std_deviation / initial_std_deviation) ** (
        1 / (num_epochs*5)
    )
    # Initialize optimizer for model parameters
    optimizer = torch.optim.Adam(
        som_model.parameters(), lr=initial_learning_rate
    )

    # Initialize lists to store output values
    learning_rates = []
    epoch_losses = []
    activations_per_epoch = []
    weights_per_epoch = []

    # Iterate over epochs
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Calculate standard deviation for current epoch
        current_std_deviation = (
            initial_std_deviation * std_deviation_decay**epoch
        )

        # Calculate learning rate for current epoch
        current_learning_rate = (
            initial_learning_rate * learning_rate_decay**epoch
        )

        # Iterate over data batches
        for batch_data in loader:
            # inputs, targets = batch_data
            inputs, targets = batch_data

            optimizer.zero_grad()

            # Forward pass through the model
            outputs = som_model(inputs, current_std_deviation)

            # Calculate loss
            radial_targets = som_model.radial(targets, current_std_deviation, as_point=True )
            som_loss_value = stm_loss(outputs, radial_targets)
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
        activations_per_epoch.append(
            np.stack(som_model.get_representation('grid'))
        )
        weights_per_epoch.append(np.stack(som_model.weights.tolist()))

    # Return output values
    return (
        learning_rates,
        epoch_losses,
        activations_per_epoch,
        weights_per_epoch,
    )

class NumpyDataset(Dataset):
    def __init__(self, n=1000, sample_groups=40, target_radius=0.90, noise_std=0.08):
        """
        Args:
            n (int): Total number of sampling points. Default is 1000.
            sample_groups (int): Number of target repetitions/samples per target. Default is 40.
            target_radius (float): Radius for the targets uniformly distributed along a circle. Default is 0.90.
            noise_std (float): Standard deviation of Gaussian noise added to inputs. Default is 0.3.
        """
        self.n = n
        self.sample_groups = sample_groups
        self.target_radius = target_radius
        self.noise_std = noise_std
        self.features, self.labels = self._generate_data()

    def _generate_data(self):
        # Generate target points uniformly distributed along the circle with radius target_radius
        target_angles = np.linspace(0, 2 * np.pi, self.sample_groups, endpoint=False)
        targets = np.vstack([self.target_radius * np.cos(target_angles), self.target_radius * np.sin(target_angles)]).T

        # Repeat each target n // sample_groups times to align
        targets = np.repeat(targets, self.n // self.sample_groups, axis=0)    

        # Generate inputs from Gaussian distribution centered at the repeated targets with std noise_std
        inputs = targets + self.noise_std * np.random.randn(self.n, 2)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        
        # Scale targets 
        targets = 10 * (targets + 0.5)

        return inputs, targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (feature, label) where feature is a tensor version of `self.features[index]` 
            and label is a tensor version of `self.labels[index]`
        """
        feature = torch.tensor(self.features[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        return feature, label

# Number of samples
n = 1000

# Number of groups in the sample
sample_groups = 40

# Desired target radius
target_radius = 0.90

# Standard deviation of noise to be added
noise_std = 0.15

# Determine the shape of the inputs
input_num = 1000  # Number of input samples
input_size = 2    # Dimensionality of each input sample

# Define the size of the output
output_size = 100

# Define scheduling variables
epochs_num = 300  # Number of epochs for training
batch_size = 50   # Number of samples in each training batch



# Create dataset
numpy_dataset = NumpyDataset(n=n, sample_groups=sample_groups, target_radius=target_radius, noise_std=noise_std)
inputs, targets = numpy_dataset.features, numpy_dataset.labels
# Create dataloader
data_loader = DataLoader(
    numpy_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

# create topological map with 100 units
topological_map = TopologicalMap(input_size, output_size)
train_som(topological_map, data_loader, num_epochs=epochs_num)

# %%

# Creating a subplot with 2 rows and 1 column, setting the figure size
fig, (ax, iax) = plt.subplots(2, 1, figsize=(4, 8))

# Extracting the topological map weights as a numpy array
weights = topological_map.weights.cpu().detach().numpy()

# Scatter plot of the weights in the first subplot (ax)
ax.scatter(weights[0], weights[1])

# Plotting the grid lines in the topological map
# Reshape the weights into a 2x10x10 grid and transpose to get rows
for row in weights.reshape(2, 10, 10).transpose(1, 2, 0):
    ax.plot(row.T[0], row.T[1], color='black')

# Reshape the weights into a 2x10x10 grid and transpose to get columns
for row in weights.reshape(2, 10, 10).transpose(2, 1, 0):
    ax.plot(row.T[0], row.T[1], color='black')

# Scatter plot of the input data points
ax.scatter(inputs[:, 0], inputs[:, 1])

# Scatter plot of the target data points in the second subplot (iax)
iax.scatter(targets[:, 0], targets[:, 1], s=60, color='red')

# Display the plots
plt.show()

torch.save(topological_map.weights, "attentional_weights")
