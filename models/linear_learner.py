# Python libraries
from datetime import date
import os
from typing import Dict, Optional

# Data science
import pandas as pd
from pandas import DataFrame

# Amazon imports
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator

# Local imports
from data_processing.data_loader import DataLoader
from exectors.sagemaker import Sagemaker

from .base import BaseModel


class AwsLinearLearner(BaseModel):
    """
    The AWS Linear Learner model.
    """

    def __init__(
        self,
        data: DataLoader,
        aws_executor: Sagemaker,
        output_path=None,
        local_save_folder="data/temp/linear_learner",
    ) -> None:
        """
        Initializes the AwsLinearLearner with data and an executor.
        This will not yet do any training or data uploading.
        """
        self.data = data
        self.executor = aws_executor
        self.local_save_folder = local_save_folder

        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = "s3://{bucket}/{prefix}/linear_learner".format(
                bucket=self.executor.bucket, prefix=self.executor.prefix
            )

        self._model = None

    def train(self, hyperparameters: Dict = {}) -> None:
        """
        Trains the model, with the data provided

        Arguments:
            hyperparameters: The hyperparameters to provide to the LinearLearner model.
                See https://sagemaker.readthedocs.io/en/stable/linear_learner.html
        """
        container = get_image_uri(self.executor.boto_session.region_name, "linear-learner")

        self._model = Estimator(
            container,
            **self.executor.default_model_kwargs,
            output_path=self.output_path,
        )
        self._model.set_hyperparameters(**hyperparameters)

        # Get the data and upload to S3
        Y_train = self.data.train_data.loc[:, self.data.output_column]
        X_train = self.data.train_data.loc[:, self.data.feature_columns]
        train_location = self._prepare_data("train", X_train, Y_train)

        Y_validation = self.data.validation_data.loc[:, self.data.output_column]
        X_validation = self.data.validation_data.loc[:, self.data.feature_columns]
        validation_location = self._prepare_data(
            "validation", X_validation, Y_validation
        )

        self._model.fit({"train": train_location, "validation": validation_location})

    def _prepare_data(
        self, data_name: str, x_data: DataFrame, y_data: Optional[DataFrame] = None
    ) -> str:
        """
        Prepares the data to use in the learner.

        Arguments:
            x_data: the features of the data
            y_data: (optional) the output of the data. Don't provide for predictions

        Returns:
            The S3 location of the data
        """
        if not os.path.exists(self.local_save_folder):
            os.makedirs(self.local_save_folder)

        temp_location = f"{self.local_save_folder}/{data_name}.csv"
        if y_data is not None:
            data = pd.concat([y_data, x_data], axis=1)
        else:
            data = x_data
        data.to_csv(temp_location, index=False, header=False)
        s3_prefix = f"{self.output_path}/input_data"
        return self.executor.upload_data(temp_location, prefix=s3_prefix)

    def load_model(self) -> None:
        """
        Load the already trained model to not have to train again.
        """
        return NotImplemented

    def predict(self, test: bool = True) -> DataFrame:
        """
        Predict based on an already trained model.

        Arguments:
            test: whether to only use the test from the data loader or to use the full data loader
        """
        return NotImplemented
