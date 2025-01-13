import torch

class Options:
    """
    A class for defining training hyperparameters and configuration options.
    """
    def __init__(self):
        self.dont_stop_training = True  # Continue training indefinitely unless a condition is met
        self.save_model = True          # Save the model after training
        self.debug = False              # Enable debug mode
        self.learning_rate = 0.0001     # Learning rate for the optimizer
        self.train_batch_size = 29      # Batch size for training
        self.test_batch_size = 200      # Batch size for testing
        self.iterations = 10000         # Number of training iterations
        self.log_interval = 20          # Interval for logging progress
        self.num_workers = 2            # Number of workers for data loading
        self.dataset_path = "./dataset/" # Path to the dataset (must end with '/')
        self.info_file_path = "info.txt" # Path to the info file
        self.device = torch.device("cpu") # Default device (CPU)

    def update_device(self):
        """
        Update the device to CUDA if available.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
