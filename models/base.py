from abc import ABC
import logging
from typing import Dict, Any, Optional

# Data science libraries
from pandas import DataFrame

# Local imports
from data_processing.data_loader import DataLoader

LOGGER = logging.getLogger(__name__)


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

    def predict(
        self, data_loader: Optional[DataLoader] = None, all_data: bool = False, **kwargs
    ) -> DataFrame:
        """
        Predict based on an already trained model.

        Arguments:
            test: whether to only use the test from the data loader or to use the full data loader
        """
        LOGGER.info("Starting the PyTorch predictions")
        self.load_predictor()
        if data_loader is not None:
            data_loader = self.data

        if all_data:
            data = data_loader.data
        else:
            data = data_loader.test_data
        X_test = data.loc[:, data_loader.feature_columns]

        # Here the magic actually happens. Implementation specific
        predictions = self.execute_prediction(data, **kwargs)

        predictions = data_loader.format_predictions(predictions, all_data=all_data)
        return predictions

    def execute_prediction(self, data: DataFrame, **kwargs) -> DataFrame:
        """
        Actually executes the predictions. Based on implementation

        Arguments:
            data: The data to predict
            **kwargs: Any other parameters for the correct execution

        Return:
            The predictions in a dataframe
        """
        return NotImplemented

    def tune(self, *args, **kwargs) -> None:
        """
        Tunes the current models with the provided hyperparameter tuning dict.
        """
        return NotImplemented
