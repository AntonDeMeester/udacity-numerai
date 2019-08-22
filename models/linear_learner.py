# Python libraries
from datetime import date
import logging
import os
from typing import Dict, Optional

# Data science
import pandas as pd
from pandas import DataFrame

# Amazon imports
from sagemaker import s3_input
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.predictor import RealTimePredictor
from sagemaker.transformer import Transformer

# Local imports
from data_processing.data_loader import DataLoader
from executors.sagemaker import Sagemaker

from .base import BaseModel

LOGGER = logging.getLogger(__name__)

class AwsLinearLearner(BaseModel):
    """
    The AWS Linear Learner model.
    """

    default_hyperparameters: Dict = {
        "predictor_type": "regressor",
        "feature_dim": 1,
        "epochs": 10,
        "loss": "auto"
    }

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
        LOGGER.info("Initializing AWS Liner learner model")

        self.data = data
        self.executor = aws_executor
        self.local_save_folder = local_save_folder
        self.model_name = None
        self.prefix = f"{self.executor.prefix}/linear_learner"
        self.input_data_prefix = f"{self.prefix}/input_data"
        self.output_data_prefix = f"{self.prefix}/output_data"

        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = "s3://{bucket}/{prefix}/linear_learner".format(
                bucket=self.executor.bucket, prefix=self.prefix
            )

        self._model: Optional[Estimator] = None
        self._transformer: Optional[Transformer] = None
        self._predictor: Optional[RealTimePredictor] = None

    def train(self, hyperparameters: Dict = {}) -> None:
        """
        Trains the model, with the data provided

        Arguments:
            hyperparameters: The hyperparameters to provide to the LinearLearner model.
                See https://sagemaker.readthedocs.io/en/stable/linear_learner.html
        """
        LOGGER.info("Starting to train model.")
        self._model = self._get_model(hyperparameters)

        # Get the data and upload to S3
        # Y_train = self.data.train_data.loc[:, self.data.output_column]
        # X_train = self.data.train_data.loc[:, self.data.feature_columns]
        # s3_input_train = self._prepare_data("train", X_train, Y_train)
        s3_input_train = s3_input('s3://sagemaker-eu-west-1-729071960169/data/input_data/train.csv', content_type="text/csv")

        # Y_validation = self.data.validation_data.loc[:, self.data.output_column]
        # X_validation = self.data.validation_data.loc[:, self.data.feature_columns]
        # s3_input_validation = self._prepare_data(
        #     "validation", X_validation, Y_validation
        # )
        s3_input_validation = s3_input('s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/validation.csv', content_type="text/csv")
        
        LOGGER.info("Starting to fit model")
        self._model.fit({"train": s3_input_train, "validation": s3_input_validation})
        LOGGER.info("Done with fitting model")

    def _get_model(self, hyperparameters: Dict = {}) -> Estimator:
        """
        Initializes the model. This can be used to train later or attach an existing model

        Arguments:
            hyperparameters: The hyperparameters for the LinearLearner model

        Returns:
            model: The initialized model
        """
        container = get_image_uri(
            self.executor.boto_session.region_name, "linear-learner"
        )
        model = Estimator(
            container,
            **self.executor.default_model_kwargs,
            output_path=self.output_path,
        )
        used_hyperparameters = self.default_hyperparameters
        used_hyperparameters["feature_dim"] = len(self.data.feature_columns)
        used_hyperparameters["output_path"] = self.output_path
        used_hyperparameters.update(hyperparameters)

        model.set_hyperparameters(**used_hyperparameters)
        return model

    def _prepare_data(
        self, data_name: str, x_data: DataFrame, y_data: Optional[DataFrame] = None, s3_input: bool = True
    ) -> s3_input:
        """
        Prepares the data to use in the learner.

        Arguments:
            x_data: the features of the data
            y_data: (optional) the output of the data. Don't provide for predictions

        Returns:
            The s3 input of the data
        """
        LOGGER.info("Preparing data for usage")
        if not os.path.exists(self.local_save_folder):
            os.makedirs(self.local_save_folder)

        temp_location = f"{self.local_save_folder}/{data_name}.csv"
        if y_data is not None:
            data = pd.concat([y_data, x_data], axis=1)
        else:
            data = x_data
        LOGGER.debug("Writing data to local machine")
        data.to_csv(temp_location, index=False, header=False)
        if s3_input:
            return self.executor.upload_data_for_model(temp_location, prefix=self.input_data_prefix, content_type="text/csv")
        return self.executor.upload_data(temp_location, prefix=self.input_data_prefix)

    def load_model(self, model_name: str) -> None:
        """
        Load the already trained model to not have to train again.

        Arguments:
            model_name: The name of the training job, as provided by AWS
        """
        LOGGER.info(f"Loading already trained model {model_name}")
        self._model = Estimator.attach(
            training_job_name=model_name,
            sagemaker_session=self.executor.session
        )

    def batch_predict(self, test: bool = True) -> DataFrame:
        """
        Predict based on an already trained model.
        Loads the existing model if it exists.

        Arguments:
            test: whether to only use the test from the data loader or to use the full data loader
        
        Returns:
            The predicted dataframe
        """
        LOGGER.info(f"Predicting new data")
        if self._transformer is None:
            self._transformer = self._get_transformer()
        
        if test:
            data = self.data.test_data
        else:
            data = self.data.data

        # Get the data and upload to S3
        X_test = data.loc[:, self.data.feature_columns]
        s3_location_test = self._prepare_data("test", X_test, s3_input=False)

        # Start the job
        self._transformer.transform(
            s3_location_test,
            content_type="text/csv",
            split_type="Line"
        )
        self._transformer.wait()

        # Download the data
        Y_test = self._load_results("test")
        Y_test.index = data.index
        Y_test.columns = data.output_column

        return Y_test

        
    def _get_transformer(self) -> Transformer:
        """
        Returns a transformer based on the current model
        """
        assert self._model is not None, (
            "Cannot create a transformer if the model is not yet set."
        )
        return self._model.transformer(
            **self.executor.default_transformer_kwargs,
            output_path=f"{self.output_path}/output_predictions"
        )

    def _load_results(self, file_name: str) -> DataFrame:
        local_file_location = self.executor.download_data(
            f"{file_name}.csv.out", 
            self.local_save_folder,
            prefix=self.output_data_prefix,
        )
        return pd.read_csv(local_file_location, header=None, index=None)
