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

    def __init__(self, data: DataLoader) -> None:
        """
        Loads the data.

        Arguments:
            data: The dataloader to use
        """
        self.data = data

    def train(self) -> None:
        """
        Trains the model, with the data provided
        """
        return NotImplemented

    def load_model(self) -> None:
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
            data_looader: The data to predict. If not provided, it will default to the local data loader.
            all_data: Whehter to predict all the data in the data loader. If false, the test data will be predicted.
        """
        if data_loader is None:
            data_loader = self.data

        if all_data:
            data = data_loader.data
        else:
            data = data_loader.test_data
        X_test = data.loc[:, data_loader.feature_columns]

        # Here the magic actually happens. Implementation specific
        predictions = self.execute_prediction(X_test, **kwargs)

        predictions = data_loader.format_predictions(predictions, all_data=all_data)
        return predictions

    def execute_prediction(self, data: DataFrame) -> DataFrame:
        """
        Actually executes the predictions. Based on implementation

        Arguments:
            data: The data to predict
            **kwargs: Any other parameters for the correct execution

        Return:
            The predictions in a dataframe
        """
        return NotImplemented

    def tune(self) -> None:
        """
        Tunes the current models with the provided hyperparameter tuning dict.
        """
        return NotImplemented
