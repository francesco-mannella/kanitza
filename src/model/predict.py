import torch
import torch.nn as nn

class Predict(nn.Module):
    def __init__(self, layer_sizes):
        super(Predict, self).__init__()
        
        layers = []  # Initialize list to store layers
        
        # Create layers based on the list of layer sizes
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add non-linear activation between intermediate layers
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
            # Add sigmoid activation to the last layer
            elif i == len(layer_sizes) - 2:
                layers.append(nn.Sigmoid())
        
        # Wrap the layers in a Sequential container
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PredictUpdater:

    def __init__(self, predict, learning_rate):

        self.optimizer = optim.Adam(
            params=stm.parameters(), lr=learning_rate
        )

        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output,  learning_modulation=1):
    
        loss = learning_modulation * self.loss(output)        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()


