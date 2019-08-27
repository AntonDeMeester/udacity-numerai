from __future__ import annotations

# Default Python libraries
from datetime import date
import logging
import typing
from typing import List, Tuple, Optional, Iterable, Dict, Any

# Data science libraries
import numpy as np
import pandas as pd
import sklearn

from pandas import DataFrame
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


class DataLoader:
    """
    This class will load and manipulate the data.
    At creation, you need to provide the data location.
    """

    index_column = "id"
    data_type_column = "data_type"
    time_column = "era"
    feature_columns = None  # Dynamically load these as they are big
    output_column = "target_kazutsugi"

    def __init__(
        self,
        local_data_location: Optional[str] = None,
        data: Optional[DataFrame] = None,
        validation_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> None:
        """
        Initialise the data loader.
        You can provide either a data locaiton or direct data

        Arguments:
            local_data_location: The location of the file to load the data from
            data: The dataframe 
        """
        assert (0 < validation_frac + test_frac < 1) and (
            validation_frac >= 0 and test_frac >= 0
        ), """
        Both validation and test fraction need to be higher or equal to zero.
        The sum must be lower than 1.
        Currently validation_frac is {} and test_frac {}
        """.format(
            validation_frac, test_frac
        )

        assert not (
            local_data_location is None and data is None
        ), """
        Both local data location and data cannot be filled in at the same time.
        """

        self.local_data_location = local_data_location
        self.validation_frac = validation_frac
        self.test_frac = test_frac
        self.feature_columns = self._get_feature_columns()

        # Allow caches to be set on the data (e.g. s3 locations)
        self.cache: Dict[str, Any] = {}

        # We will only load the data when required
        self._data = data

        # Initialize several subsets of data to load later
        self._train_data = None
        self._validation_data = None
        self._test_data = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the data when requested. 
        The property only loads the data when it's necessary to safe memory.
        """
        if self._data is not None:
            return self._data
        self._data = pd.read_csv(
            self.local_data_location, index_col=self.index_column, low_memory=False
        )
        return self._data

    @property
    def train_data(self) -> pd.DataFrame:
        """
        Returns the train data when requested. 
        The property only loads the data when it's necessary to safe memory.
        """
        if self._train_data is not None:
            return self._train_data
        self._split_train_validation_test(self.validation_frac, self.test_frac)
        return self._train_data

    @property
    def validation_data(self) -> pd.DataFrame:
        """
        Returns the validation data when requested. 
        The property only loads the data when it's necessary to safe memory.
        """
        if self._validation_data is not None:
            return self._validation_data
        self._split_train_validation_test(self.validation_frac, self.test_frac)
        return self._validation_data

    @property
    def test_data(self) -> pd.DataFrame:
        """
        Returns the test data when requested. 
        The property only loads the data when it's necessary to safe memory.
        """
        if self._test_data is not None:
            return self._test_data
        self._split_train_validation_test(self.validation_frac, self.test_frac)
        return self._test_data

    @classmethod
    def _get_feature_columns(self) -> List[str]:
        """
        Defines all the feature columns for Numerai.
        """
        # NOTE: When making this general, maybe let this be overwritable?
        features = []
        column_ranges: List[Dict] = [
            {"prefix": "feature_intelligence", "range": range(1, 12 + 1)},
            {"prefix": "feature_charisma", "range": range(1, 86 + 1)},
            {"prefix": "feature_strength", "range": range(1, 38 + 1)},
            {"prefix": "feature_dexterity", "range": range(1, 14 + 1)},
            {"prefix": "feature_constitution", "range": range(1, 114 + 1)},
            {"prefix": "feature_wisdom", "range": range(1, 46 + 1)},
        ]
        for column in column_ranges:
            for index in column["range"]:
                features.append(f"{column['prefix']}{str(index)}")
        return features

    def _split_train_validation_test(
        self, validation_frac: float, test_frac: float
    ) -> None:
        """
        Splits the internal data in a train, validation and test dataframe, based on the parameters.
        validation_frac and test_frac must be larger or equal to 0, the sum must be between 0 and 1, not inclusive
        
        Arguments:
            validation_frac: The fraction for the validation set
            test_frac: The fraction for the test test
            
        Returns:
            None
        """
        LOGGER.info("Splitting the train, validation and test data")
        assert 0 < validation_frac + test_frac < 1
        assert validation_frac >= 0 and test_frac >= 0
        test_val_frac = validation_frac + test_frac
        train, validation = train_test_split(
            self.data, test_size=test_val_frac, random_state=512
        )  # Set the random state so we can get consistent results
        LOGGER.info("Done splitting train vs validation-test")

        validation_over_test_frac = validation_frac / test_val_frac
        if validation_over_test_frac == 1:
            test = None
        elif validation_over_test_frac == 0:
            test = validation
            validation = None
        else:
            validation, test = train_test_split(
                validation, train_size=validation_over_test_frac, random_state=512
            )  # Set the random state so we can get consistent results
        LOGGER.info("Done splitting validation and test")

        self._train_data = train
        self._validation_data = validation
        self._test_data = test

    def split_in_batches(self, number_of_batches: int) -> List[DataLoader]:
        """
        Splits the current DataLoader in a number of DataLoaders with the split data set.

        Arguments:
            number_of_batches: The number of split batches to return

        Returns:
            a list of DataLoader objects: The combined data of the DataLoaders is equal to the data of the current DataLoader.
                Then number of objects is equal to number_of_batches
        """
        LOGGER.info("Splitting the data in batches")

        # Suffle the df first so it's actually random
        intermediate = self.data.sample(frac=1)
        number_of_rows = intermediate.shape[0]
        list_of_data_loaders = []

        for i in range(number_of_batches):
            start_index = (i * number_of_rows) // number_of_batches
            end_index = ((i + 1) * number_of_rows) // number_of_batches
            batch = intermediate.iloc[start_index:end_index]
            list_of_data_loaders.append(
                DataLoader(
                    data=batch,
                    validation_frac=self.validation_frac,
                    test_frac=self.test_frac,
                )
            )

        LOGGER.info("Done splitting the data in batches")

        return list_of_data_loaders

    def add_to_cache(self, data_type: str, data_name: str, data: Any) -> None:
        """
        Adds data to the cache
        For example could be used to cache s3 locations of the uploaded data instead of 
        adding the data multiple time to the data.

        Arguments:
            data_type: The type of data. One of 'local', 's3'.
            data_name: The name of the data, e.g. validation, X_test...
            data: The data to cache.
        """
        if data_type not in self.cache:
            self.cache[data_type] = {}

        self.cache[data_type][data_name] = data

    def get_from_cache(self, data_type: str, data_name: str) -> Optional[Any]:
        """
        Gets data from cache.
        The data should have been set with add_to_cache beforehand.
        Returns None is it is not present.

        Arguments:
            data_type: The type of data. One of 'local', 's3_flat', 's3_input'
            data_name: The name of the data, e.g. validation, X_test...
        
        Returns:
            data: the cached data
        """
        try:
            return self.cache[data_type][data_name]
        except KeyError:
            return None

    def score_data(self, Y_pred: DataFrame, all_data: bool = False) -> float:
        """
        Scores the data versus the predictions.
        For numerai, corretation coefficient is used.

        Arguments:
            Y_pred: the predicted values
            all_data: Whether to use the complete dataset to compare to, or just the test set

        Returns:
            The scoring metric (correlation coefficient)
        """
        if all_data:
            Y_labels = self.data
        else:
            Y_labels = self.test_data
        Y_labels = Y_labels.loc[:, self.output_column]
        metric = self.score_correlation(Y_labels, Y_pred)
        return metric

    def score_correlation(self, labels: DataFrame, prediction: DataFrame) -> float:
        """
        Scores the correlation as defined by the Numerai tournament rules.

        Arguments:
            labels: The real labels of the output
            prediction: The predicted labels

        Returns:
            The correlation coefficient
        """
        ranked_prediction = prediction.rank(pct=True, method="first")
        return np.corrcoef(labels, ranked_prediction, rowvar=False)[0, 1]
