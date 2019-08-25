from abc import ABC

# Data science libraries
from pandas import DataFrame

# Local imports
from data_processing.data_loader import DataLoader


class BaseModel(ABC):
    """
    The base model to be used in this project.

    This is an abstract class. Please subclass it to implement each method.
    """

    def __init__(self, data: DataLoader, *args, **kwargs) -> None:
        """
        Loads the data.

        Arguments:
            data: The dataloader to use
        """
        self.data = data

    def train(self, *args, **kwargs) -> None:
        """
        Trains the model, with the data provided
        """
        return NotImplemented

    def load_model(self, *args, **kwargs) -> None:
        """
        Load the already trained model to not have to train again.
        """
        return NotImplemented

    def predict(self, all_date: bool = False, *args, **kwargs) -> DataFrame:
        """
        Predict based on an already trained model.

        Arguments:
            test: whether to only use the test from the data loader or to use the full data loader
        """
        return NotImplemented
