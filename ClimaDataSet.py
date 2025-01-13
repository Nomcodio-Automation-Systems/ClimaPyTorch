import torch
from torch.utils.data import Dataset

class ClimateDataset(Dataset):
    """
    A PyTorch Dataset class for loading climate data.

    This class expects the data to be provided as a list of pairs of tensors,
    where each pair consists of (inputs, targets).

    Args:
        data (list of tuple): A list of tuples, where each tuple contains
                              an input tensor and a target tensor.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """
        Retrieve a single data point from the dataset.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the input tensor and target tensor.
        """
        inputs = self.data[index][0]
        targets = self.data[index][1]

        return inputs.clone(), targets.clone()

    def __len__(self):
        """
        Get the size of the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self.data)
