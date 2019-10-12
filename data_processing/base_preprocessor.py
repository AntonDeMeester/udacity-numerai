# Python
from abc import ABC
import logging
from typing import Any, Iterable, Union

# Data science
from pandas import DataFrame

# Local
from .data_loader import DataLoader

LOGGER = logging.getLogger(__name__)

all_colmun_name = "__all__"


class BasePreprocesor(ABC):
    """
    Template for any data preprocessor
    """

    def __init__(
        self, data: DataLoader, columns: Union[Iterable[str], Iterable[int], str]
    ):
        self.initial_data = data
        if columns == all_colmun_name:
            self.columns = self.initial_data.feature_columns
        else:
            self.columns = columns
        self._model: Any = None

    def process_data(self, data: DataFrame = None) -> DataFrame:
        """
        Main method to process the data.
        Input: 
            data: DataFrame to process. If not provided, will default to its own data

        Output:
            DataFrame with the processed data
        """
        LOGGER.info("Starting with processing data")
        processed_data = None
        if data is None:
            data = self.initial_data.data
        data = self.select_data(data)

        if self._model is None:
            processed_data = self.train_processor(data)

        if processed_data is None:
            processed_data = self._process_data(data)

        LOGGER.info("Done with processing data")
        return processed_data

    def select_data(self, data: DataFrame) -> DataFrame:
        return data.loc[:, self.columns]

    def train_processor(self, data: DataFrame) -> Union[DataFrame, None]:
        LOGGER.info("Starting training the data preprocessor")
        formatted_train_data = self.format_train_input_data(data)
        output_data = self.execute_training(formatted_train_data)
        if output_data is not None:
            output_data = self.format_train_output_data(output_data, data)
        LOGGER.info("Done with training the data preprocessor")
        return output_data

    def format_train_input_data(self, data: DataFrame) -> Any:
        return self.format_input_data(data)

    def execute_training(self, formatted_data: Any) -> Any:
        raise NotImplementedError()

    def format_train_output_data(
        self, output_data: Any, initial_data: DataFrame
    ) -> DataFrame:
        return self.format_output_data(output_data, initial_data)

    def _process_data(self, data: DataFrame):
        formatted_train_data = self.format_input_data(data)
        output_data = self.execute_processing(formatted_train_data)
        formatted_output_data = self.format_output_data(output_data, data)
        return formatted_output_data

    def format_input_data(self, data: DataFrame) -> Any:
        return data

    def execute_processing(self, formatted_data: Any) -> Any:
        raise NotImplementedError()

    def format_output_data(
        self, output_data: Any, initial_data: DataFrame
    ) -> DataFrame:
        return output_data

    def load_model(self, model_location: Any):
        raise NotImplementedError()
