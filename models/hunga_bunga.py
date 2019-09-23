# General python imports
from typing import Union, Optional, Dict, Any

# General data science imports
from hunga_bunga import (
    HungaBungaClassifier as HBClassifier,
    HungaBungaRegressor as HBRegressor,
)
import numpy as np
import pandas as pd
from pandas import DataFrame

# Local imports
from data_processing.data_loader import DataLoader

from .base import BaseModel


class HungaBungaBase(BaseModel):
    """
    Te Hunga Bunga model combines all SKLearn models and selects the best one.
    It's not allowed to select a validation set manually.

    Not recommended for large sample sizes as it's quite taxing on your own system.
    """

    name = "hunga-bunga"

    def __init__(self, data: DataLoader, local_save_folder: str = None) -> None:
        """
        Loads the data.

        Arguments:
            data: The dataloader to use
        """
        super().__init__(data)
        if local_save_folder is not None:
            self.local_save_folder = local_save_folder
        else:
            self.local_save_folder = f"data/temp/{self.name}"

        self._model: Union[HBClassifier, HBRegressor] = NotImplemented

    def train(self) -> None:
        """
        Trains the model, with the data provided
        """
        all_data = pd.concat([self.data.train_data, self.data.validation_data])
        X = all_data.loc[:, self.data.feature_columns].values
        Y = all_data.loc[:, self.data.output_column].values
        all_data = None
        self._model.fit(X, Y)

    def execute_prediction(self, data: DataFrame, name: str = "test") -> DataFrame:
        """
        Actually executes the predictions.
        """
        predictions = self._model.predict(data.values)
        return DataFrame(predictions)

    def tune(self):
        """
        For this model, tuning is the same as training
        """
        return self.train()


class HungaBungaClassifier(HungaBungaBase):
    """
    Classifier model for Hunga Bunga
    """

    name = "hunga-bunga-classifier"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model: HBClassifier = HBClassifier()


class HungaBungaRegressor(HungaBungaBase):
    """
    Regressor model for Hunga Bunga
    """

    name = "hunga-bunga-classifier"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model: HBRegressor = HBRegressor()
