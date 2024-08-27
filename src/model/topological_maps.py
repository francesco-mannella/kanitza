import torch
import torch.optim as optim
import math
import numpy as np


class DimensionalityError(Exception):
    def __str__(self):
        return "Dimensionality of the output must be 1D or 2D."


class RadialBasis:
    """
    This code creates a radial grid in either 1D or 2D based on a given
    centroid. It can be used to generate a grid of points around a central
    point.
    """
    def __init__(self, size, dims):
        """
        This function creates a radial grid with a given size and dimensionality. 

        Args:
            size (int): The size of the radial grid.
            dims (int): The dimensionality of the grid (1 for 1D, 2 for 2D).

        """
        self.dims = dims
        self.size = size

        if self.dims == 1:
            self.grid = torch.arange(self.size)
            self.side = self.size
        elif self.dims == 2:
            self.side = int(math.sqrt(self.size))
            if self.side**2 != self.size:
                raise "Dinensions must be equal"
            t =  torch.arange(self.side)
            meshgrids = torch.meshgrid(t, t, indexing="ij")
            self.grid = torch.stack([x.reshape(-1) for x in meshgrids]).T
        else:
            raise DimensionalityError()
        self.grid = self.grid.unsqueeze(dim=0).float()

    def __call__(self, index, std, as_point=False):
        """
        Args:
            index (int): indicates the point at the center of the Gaussian on the flattened grid, arranged in rows.
            std (float): The standard deviation of the function.
            as_point (bool, optional): Whether to treat index as a point or not. Defaults to False.

        Returns:
            The result of the function call.
        """
        
        if self.dims == 1:
            x = index.unsqueeze(dim=-1)
            dists = self.grid - x
        elif self.dims == 2:
            if as_point:
                index = index.reshape(-1, 2)
                col, row = index[:, 0], index[:, 1]
            else:
                row = index // self.side
                col = index % self.side
            x = torch.stack([row, col]).T
            x = x.unsqueeze(dim=1)
            dists = torch.norm(
                    self.grid - x, 
                    dim=-1)
        output = torch.exp(-0.5 * (std**-2) * dists**2)
        output /= output.sum(dim=-1).unsqueeze(dim=-1)

        return output

class TopologicalMap(torch.nn.Module):
    """
    A neural network designed to represent a topological map. The nodes are
    connected in a way that reflects the topology of the data, allowing the network
    to recognize patterns and make decisions based on those patterns. 
    """
    def __init__(self, input_size, output_size, output_dims=2, parameters = None):
        """

        Args:
            input_size (int): The number of inputs for the network.
            output_size (int): The number of outputs for the network.
            output_dims (int, optional): The number of dimensions for the output. Defaults to 2.
            parameters (nparray): The array of initial weights. default is None
        """
        
        super(TopologicalMap, self).__init__()

        if parameters is None:
            weights = torch.empty(input_size, output_size)
            torch.nn.init.xavier_normal_(weights)
            self.weights = torch.nn.Parameter(weights, requires_grad=True)
        else:
            parameters = torch.tensor(parameters).float()
            self.weights = torch.nn.Parameter(parameters, requires_grad=True)

        self.input_size = input_size
        self.output_size = output_size
        self.output_dims = output_dims
        self.radial = RadialBasis(output_size, output_dims)
        self.std_init = (self.output_size if output_dims == 1 
                         else int(math.sqrt(self.output_size)))
        self.curr_std = self.std_init
        self.bmu = None
        self.side = None if output_dims == 1 else int(math.sqrt(self.output_size))

    def forward(self, x: torch.Tensor, std: float) -> torch.Tensor:
        """
        Computes the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        diffs = self.weights.unsqueeze(dim=0) - x.unsqueeze(dim=-1)
        norms = torch.norm(diffs, dim=1)
        norms2 = torch.pow(norms, 2)
        self.bmu = torch.argmin(norms, dim=-1).detach()
        phi = self.radial(self.bmu, std)
        self.curr_std = std
        self.norms = norms

        return norms2*phi

    def get_representation(self, rtype="point"):
        """Returns the representation of the best matching unit (BMU) based on
        the specified representation type.

        Args:
            rtype (str, optional): The representation type to be returned. Valid
                values are: "point" (default) and "grid".

        Returns:
            torch.Tensor or None: The representation of the BMU. Returns
            None if the BMU is not available.
        """
        if self.bmu is not None:
            if rtype == "point":
                if self.output_dims == 1:
                    return self.bmu.tolist() 

                elif self.output_dims == 2:
                    row = self.bmu // self.side
                    col = self.bmu % self.side
                    return torch.stack([row, col]).T.float() 

            elif rtype == "grid":
                std = self.curr_std
                phi = self.radial(self.bmu, std)
                return phi
        else:
            return None

    def backward(self, point, std=None):
        """
        Computes the backward pass of the given point.

        Args:
            point (int): The point to compute the backward pass for.
            std (float, optional): The standard deviation to use for the backward pass. Defaults to None.

        Returns:
            float: The result of the backward pass.
        """

        if std is None: std = self.curr_std
        phi = self.radial(point, std, as_point=True)
        output = torch.matmul(phi, self.weights.T)
        return  output


def som_loss(output):
    """Computes the loss for reproducing a Self-Organizing Map.

    Args:
        output (array): Output of the SOM network.

    Returns:
        float: Loss value.
    """
    return 0.5*output.mean()

def stm_loss(output, target):
    """Calculates the loss for reproducing a Supervised Topological Map.

    Args:
        output (torch.Tensor): Output of the model.
        target (torch.Tensor): Target of the model.

    Returns:
        torch.Tensor: Loss value.
    """
    filtered = output * target
    return 0.5*filtered.mean()


class SOMUpdater:

    def __init__(self, stm, learning_rate):

        self.optimizer = optim.Adam(
            params=stm.parameters(), lr=learning_rate
        )

        self.loss = som_loss

    def __call__(self, output,  learning_modulation):
    
        loss = learning_modulation * self.loss(output)        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

class STMUpdater:

    def __init__(self, stm, learning_rate):

        self.optimizer = optim.Adam(
            params=stm.parameters(), lr=learning_rate
        )

        self.loss = stm_loss

    def __call__(self, output, target, learning_modulation):
    
        loss = learning_modulation * self.loss(output, target)        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
