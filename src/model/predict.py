import torch
import torch.nn as nn
import torch.optim as optim


class Predictor(torch.nn.Module):
    def __init__(self, input_dim: int):
        """
        Initializes the Predictor module.

        Args:
            input_dim (int): The input dimension of the module.
        """
        super(Predictor, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass on the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.layer1(x)
        out = self.sigmoid(out)
        return out


class PredictorUpdater:
    """
    A class for updating the predictor model using backpropagation.
    """

    def __init__(
        self,
        predictor: nn.Module,
        learning_rate: float,
    ):
        """
        Initializes the PredictorUpdater object.

        Args:
        - predictor: The predictor model to be updated.
        - learning_rate: The learning rate for the optimizer.
        - params: the object of simulation parameters
        """
        self.optimizer = optim.Adam(
            params=predictor.parameters(), lr=learning_rate
        )

    def losses(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared_error = (x - target) ** 2
        row_wise_losses = torch.mean(squared_error, dim=1).reshape(-1, 1)
        return row_wise_losses

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        learning_modulation: float,
    ) -> None:
        """
        Performs the update step.

        Args:
        - output: The predicted output from the predictor model.
        - target: The target output.
        - learning_modulation: The modulation factor for adjusting
          the learning rate.

        Returns:
        - None
        """
        loss = torch.mean(learning_modulation * self.losses(output, target))
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
