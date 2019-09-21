# Python imports
import logging
from typing import Any, Dict, Iterable, Callable, Optional, List

# Data science imports
import pandas as pd

# Local imports
from data_processing.data_loader import DataLoader
from models.combiner import Combiner
from .base import BaseModel

LOGGER = logging.getLogger(__name__)


class MetaModel(BaseModel):
    """
    A meta model combines other models.
    Basic implementation is the same as a normal model, with train and predict models.
    """

    def __init__(self, data, models: Iterable[BaseModel], combiner: Combiner):
        """
        Initialises a MetaModel.
        Requires data, a list of models and a combination function to opmtimse.

        Arguments:
            * data: A DataLoader, like with any model
            * models: A list or any other iterable of BaseModels. This can include other meta models.
            * combiner: A BaseModel to combine the different results. 
        """
        assert models, "The list of models cannot be empty"
        assert combiner, "You need to provide a Combiner"
        super().__init__(data)

        self.models: Iterable[BaseModel] = models
        self.combiner: Combiner = combiner
        self.model_weights: List[float] = []

    def train(self):
        """
        Trains each of the models after each other.
        """
        LOGGER.info("Start training the meta model")
        for model in self.models:
            model.train()
        LOGGER.info("Done with training the meta model")

        self.calculate_weights()

    def calculate_weights(self):
        """
        Calculate the weight for each model.
        """
        # Use test data to get the predictions to combine
        predictions: List[pd.DataFrame] = []
        combine_data_features = self.data.test_data.loc[:, self.data.feature_columns]
        combine_data_labels = self.data.test_data.loc[:, self.data.output_column]
        for index, model in enumerate(self.models):
            predictions.append(model.execute_prediction(combine_data_features))

        self.model_weights = self.combiner.combine(combine_data_labels, predictions)

    def execute_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the prediction for each of the models.
        Then combines the predictions with the trained model weights.

        Arguments:
            data: The data to predict       
        """
        LOGGER.info("Predicting the data of the meta model")
        predictions = None
        for index, model in enumerate(self.models):
            Y_test = model.execute_prediction(data)
            if predictions is None:
                predictions = Y_test * self.model_weights[index]
            else:
                predictions += Y_test * self.model_weights[index]

        LOGGER.info("Done with the predictions of the meta model")
        return predictions

    def tune(self):
        """
        Tries to tune each model. If it doesn't exist, it will just train it.
        """
        LOGGER.info("Start tuning the meta model")
        for model in self.models:
            try:
                return_value = model.tune()
                if return_value is NotImplemented:
                    model.train()
            except NotImplementedError:
                model.train()

        self.calculate_weights()
        LOGGER.info("Done with tuning the meta model")
