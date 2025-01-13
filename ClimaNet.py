import torch
import torch.nn as nn
import torch.nn.functional as F

class ClimaNet(nn.Module):
    """
    A neural network model definition using PyTorch for climate modeling.

    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the first hidden layer.
        hidden_size2 (int): Number of neurons in the second hidden layer.
        hidden_size3 (int): Number of neurons in the third hidden layer.
        output_size (int): Number of output features.
    """
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, output_size):
        super(ClimaNet, self).__init__()

        # Initialize the layers of the network
        self.i2h = nn.Linear(input_size, hidden_size)       # Input to first hidden layer
        self.h2h = nn.Linear(hidden_size, hidden_size2)     # First to second hidden layer
        self.h2h2 = nn.Linear(hidden_size2, hidden_size3)   # Second to third hidden layer
        self.h2h3 = nn.Linear(hidden_size3, hidden_size3)   # Third hidden layer self-loop
        self.h2o = nn.Linear(hidden_size3, output_size)     # Final hidden layer to output layer

    def forward(self, x1):
        """
        Forward pass defines how data flows through the network.

        Args:
            x1 (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass input through the network with ReLU activations between layers
        xm = F.relu(self.i2h(x1))      # Input to first hidden layer
        xm2 = F.relu(self.h2h(xm))    # First to second hidden layer
        xm3 = F.relu(self.h2h2(xm2))  # Second to third hidden layer
        xo = F.relu(self.h2h3(xm3))   # Third hidden layer self-loop
        output = self.h2o(xo)         # Final layer produces output
        return output
