# Python libraries
from abc import ABC
import logging
from typing import Optional

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
    default_model_kwargs = {
        'framework_version': '1.0.0',
    }
    train_entry_point: str = "train.py"
    predict_entry_point: str = "predict.py"
    source_directory: str = NotImplemented

    def __init__(
        self,
        data: DataLoader,
        aws_executor: Sagemaker,
        output_path: Optional[str] = None,
        local_save_folder: Optional[str] = None,
    ) -> None:
        super().__init__(data, aws_executor, output_path, local_save_folder)

        self._estimator: Optional[PyTorch] = None
        self._model: Optional[PyTorchModel] = None
        self._predictor: Optional[PyTorchPredictor] = None

    def train(self, *args, **kwargs) -> None:
        """
        Trains the model, with the data provided
        """
        return NotImplemented

    def load_estimator(self, training_job_name: str) -> None:
        """
        Load the already trained model to not have to train again.

        Arguments:
            model_name: The name of the training job, as provided by AWS
        """
        LOGGER.info(f"Loading already trained pytorch training job: {training_job_name}")
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
                model_location = f"s3://{self.executor.bucket}/{self.output_path}/model.tar.gz"
        LOGGER.info(f"Loading already created pytorch model {model_location}")

        self._model = PyTorchModel.attach(
            model_data=model_location,
            role=self.executor.role,
            entry_point=self.predict_entry_point,
            source_directory=self.source_directory,
            sagemaker_session=self.executor.session,
            **self.default_model_kwargs,
        )

    def load_predictor(self) -> None:
        """
        Loads the predictor from the loaded model.
        If no model is present, it will load it.

        WARNING: a predictor costs money for the time it is online. 
        Make sure to always take it down.
        """
        if self._model is None:
            self.load_model()
        
        LOGGER.info("Deploying the predictor")
        self._predictor = self._model.deploy(**self.executor.default_deploy_kwargs)
        LOGGER.warn("Don't forgot to delete the predicion endpoint")

    def delete_endpoint(self) -> None:
        LOGGER.info("Deleting the pytorch endpoint")
        if self._predictor is not None:
            self._predictor.delete_endpoint()

    def execute_prediction(self, X_test: DataFrame) -> DataFrame:
        LOGGER.info("Sending the data to predict to the model")
        predictions = self._predictor.predict(data.values)
        predictions = DataFrame(predictions)
        LOGGER.info("Got the predictions")

        self.delete_endpoint()
        return predictions

    def tune(self):
        return NotImplemented