# Python libraries
from abc import ABC
import logging
import math
from typing import Optional, Dict, Any

# Data science
import pandas as pd
from pandas import DataFrame

# Amazon imports
from sagemaker.pytorch import PyTorch, PyTorchModel, PyTorchPredictor

# Local imports
from data_processing.data_loader import DataLoader
from executors.sagemaker import Sagemaker

from .aws_base import AwsBase

LOGGER = logging.getLogger(__name__)


class AwsPytorch(AwsBase, ABC):
    default_model_kwargs = {"framework_version": "1.0.0"}
    train_entry_point: str = "train.py"
    predict_entry_point: str = "predict.py"
    source_directory: str = "models/pytorch"
    name: str = "pytorch"

    max_prediction_size = 5e6

    def __init__(
        self,
        data: DataLoader,
        aws_executor: Sagemaker,
        output_path: Optional[str] = None,
        local_save_folder: Optional[str] = None,
    ) -> None:
        super().__init__(data, aws_executor, output_path, local_save_folder)

        self._estimator: PyTorch = None
        self._model: PyTorchModel = None
        self._predictor: PyTorchPredictor = None

    def train(self, hyperparameters: Dict[str, Any] = {}) -> None:
        """
        Trains the model, with the data provided
        """
        LOGGER.info("Starting to train model.")
        model = self._get_model(hyperparameters)

        # Get the data and upload to S3
        Y_train = self.data.train_data.loc[:, self.data.output_column]
        X_train = self.data.train_data.loc[:, self.data.feature_columns]
        s3_train_data = self._prepare_data(
            "train", X_train, Y_train, s3_input_type=False
        )

        Y_validation = self.data.validation_data.loc[:, self.data.output_column]
        X_validation = self.data.validation_data.loc[:, self.data.feature_columns]
        s3_validation_data = self._prepare_data(
            "validation", X_validation, Y_validation, s3_input_type=False
        )

        LOGGER.info("Starting to fit model")
        self._model.fit({"train": s3_train_data, "validation": s3_validation_data})
        LOGGER.info("Done with fitting model")

    def _get_model(self, hyperparameters):
        if self._model is not None:
            return self._model

        used_hyperparameters = {**self.default_hyperparameters, **hyperparameters}
        self._model = PyTorch(
            entry_point=self.train_entry_point,
            source_dir=self.source_directory,
            hyperparameters=used_hyperparameters,
            **self.default_model_kwargs,
            **self.executor.default_model_kwargs,
        )
        return self._model

    def load_estimator(self, training_job_name: str) -> None:
        """
        Load the already trained model to not have to train again.

        Arguments:
            model_name: The name of the training job, as provided by AWS
        """
        LOGGER.info(
            f"Loading already trained pytorch training job: {training_job_name}"
        )
        self._estimator = PyTorch.attach(
            training_job_name=training_job_name, sagemaker_session=self.executor.session
        )

    def load_model(self, model_location: Optional[str] = None) -> None:
        """
        Load the already trained model to not have to train again.
        If model_location is not provided, it will first try to see if the estimator
        has model data, else it will try a default model name.

        Arguments:
            model_location: The location of the model on S3
        """
        if model_location is None:
            if self._estimator is not None:
                model_location = self._estimator.model_data
            else:
                model_location = (
                    f"s3://{self.executor.bucket}/{self.output_path}/model.tar.gz"
                )
        LOGGER.info(f"Loading already created pytorch model {model_location}")

        self._model: PyTorchModel = PyTorchModel(
            model_data=model_location,
            role=self.executor.role,
            entry_point=self.predict_entry_point,
            source_dir=self.source_directory,
            sagemaker_session=self.executor.session,
            **self.default_model_kwargs,
        )

    def load_predictor(self, predictor_name: str = None) -> None:
        """
        Loads the predictor from the loaded model.
        If no model is present, it will load it.

        WARNING: a predictor costs money for the time it is online. 
        Make sure to always take it down.
        """
        if predictor_name is not None:
            self._predictor = PyTorchPredictor(
                predictor_name, sagemaker_session=self.executor.session
            )

        if self._predictor is not None:
            return self._predictor

        if self._model is None:
            self.load_model()

        LOGGER.info("Deploying the predictor")
        self._predictor = self._model.deploy(**self.executor.default_deploy_kwargs)
        LOGGER.warn("Don't forget to delete the predicion endpoint")

    def delete_endpoint(self) -> None:
        """
        Deletes the endpoint.
        """
        LOGGER.info("Deleting the pytorch endpoint")
        if self._predictor is not None:
            self._predictor.delete_endpoint()

    def execute_prediction(self, X_test: DataFrame) -> DataFrame:
        """
        Executes the prediction. 
        Loads and also deletes the endpoint.
        Splits the data in separate batches so they can be provided to the predictor.

        Arguments:
            X_test: the dataframe to predict

        Returns:
            Y_test: The predictions
        """
        try:
            LOGGER.info("Starting the PyTorch predictions")
            self.load_predictor()

            # Split in batches that AWS accepts. Divide by 2 for good measure
            no_batches = math.ceil(
                X_test.values.nbytes / (self.max_prediction_size / 2)
            )
            batches = self.split_in_batches(X_test, no_batches)
            prediction_list = []

            LOGGER.info("Sending the data to predict to the model")
            for batch in batches:
                predictions = self._predictor.predict(batch.values)
                prediction_list.append(DataFrame(predictions))

            Y_test = pd.concat(prediction_list, axis=0, ignore_index=True)
            LOGGER.info("Got the predictions")
        finally:
            self.delete_endpoint()
            LOGGER.info("Deleting the endpoint")
            pass
        return Y_test

    def tune(self):
        return NotImplemented

    def split_in_batches(self, data: DataFrame, number_of_batches):
        """
        Splits a dataframe in a number of batches.
        Lambda required the payload to be a maximum size, so a split needs to be made

        Arguments:
            data: Dataframe to split
            number_of_batches: The number of batches to split in

        Returns:
            A list of dataframes that combined represent the data.
        """
        LOGGER.info("Splitting the data in batches")

        number_of_rows = data.shape[0]
        list_of_dfs = []

        for i in range(number_of_batches):
            start_index = (i * number_of_rows) // number_of_batches
            end_index = ((i + 1) * number_of_rows) // number_of_batches
            batch = data.iloc[start_index:end_index]
            list_of_dfs.append(batch)

        LOGGER.info("Done splitting the data in batches")

        return list_of_dfs
