# Python libraries
import json
import logging
import os
from typing import Optional, Iterable, Union

# Data science
import pandas as pd
from pandas import DataFrame

# Amazon imports
from sagemaker import PCA, PCAModel, s3_input
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator, Transformer

# Local imports
from executors.sagemaker import Sagemaker

from .base_preprocessor import BasePreprocesor, all_colmun_name
from .data_loader import DataLoader

LOGGER = logging.getLogger(__name__)


class AwsPCA(BasePreprocesor):
    """
    Amazon PCA converter
    """

    name = "aws-pca"
    algorithm_mode = "randomized"  # Either Randomised or regular
    container_name = "pca"

    def __init__(
        self,
        data: DataLoader,
        aws_executor: Sagemaker,
        columns: Union[Iterable[str], Iterable[int], str],
        number_of_components: Union[int, float],
        output_path: Optional[str] = None,
        local_save_folder: Optional[str] = None,
    ):
        """
        Initialises the Amazon PCA data preprocessors

        Arguments:
            data: The initial data which to convert
            execute: The Sagemaker instance
            columns: Which columns to perform PCA on. Either an iterable of column names,
                an interable of indexes, or __all__ for all columns
            number_of_components: The number of output components. 
                If it's a integer, this is taken
                If it's a float and smaller than 1, this is used as a fraction, cast to int
            output_path: The output path on S3
            local_save_folder: Where to store the results locally
        """
        super().__init__(data, columns)
        self.executor = aws_executor
        if number_of_components < 1:
            self.number_of_components = int(len(self.columns) * number_of_components)
        else:
            self.number_of_components = int(number_of_components)

        self.prefix = f"{self.executor.prefix}/{self.name}"
        self.input_data_prefix = f"{self.prefix}/input_data"
        self.output_data_prefix = f"{self.prefix}/output_data"

        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = "s3://{bucket}/{prefix}".format(
                bucket=self.executor.bucket, prefix=self.output_data_prefix
            )
        if local_save_folder is not None:
            self.local_save_folder = local_save_folder
        else:
            self.local_save_folder = f"data/temp/{self.name}"

    def format_train_input_data(self, data: DataFrame) -> str:
        # We need to add a dummy output column to the data for some reason
        with_y_data = DataFrame(0, index=data.index, columns=["Y"])
        with_y_data = pd.concat([with_y_data, data], axis=1)
        input_data_location = self.upload_to_s3(with_y_data, "train")
        return s3_input(input_data_location, content_type="text/csv")

    def execute_training(self, formatted_data: str) -> None:
        LOGGER.info("Training the Aws PCA model")
        pca = self.initialise_model()
        self._model = pca.fit({"train": formatted_data})
        LOGGER.info("Done with training the Aws PCA model")
        return None

    def initialise_model(self) -> Estimator:
        container = get_image_uri(
            self.executor.boto_session.region_name, self.container_name
        )
        pca = Estimator(
            container,
            **self.executor.default_model_kwargs,
            output_path=self.output_path,
        )
        pca.set_hyperparameters(
            num_components=self.number_of_components,
            algorithm_mode=self.algorithm_mode,
            mini_batch_size=500,
            subtract_mean=False,
            feature_dim=len(self.columns),
        )
        return pca

    def format_input_data(self, data: DataFrame) -> str:
        input_data_location = self.upload_to_s3(data, "process")
        return input_data_location

    def execute_processing(self, formatted_data: str) -> str:
        LOGGER.info("Starting with processing data in the Aws PCA ")
        transformer = self.initialise_transformer()
        transformer.transform(
            formatted_data, content_type="text/csv", split_type="Line"
        )
        transformer.wait()
        LOGGER.info("Done with with processing data in the Aws PCA ")

        # TODO parametrise this?
        return "process.csv.out"

    def format_output_data(
        self, output_data: str, initial_data: DataFrame
    ) -> DataFrame:
        LOGGER.info("Downloading output data to local machine")
        # Download from S3
        local_file_location = self.executor.download_data(
            output_data, self.local_save_folder, prefix=self.output_data_prefix
        )
        LOGGER.info("Done with downloading output data to local machine")

        # Create dataframe
        data_list = []
        with open(local_file_location, "r") as data_file:
            for line in data_file:
                parsed_line = json.loads(line)
                data_list.append(parsed_line["projection"])

        df = DataFrame(data_list)
        df.index = initial_data.index

        return df

    def initialise_transformer(self) -> Transformer:
        if self._model is None:
            raise ValueError("Model not created yet when creating a transformer")
        transformer = self._model.transformer(
            **self.executor.default_transformer_kwargs,
            output_path=f"{self.output_path}",
        )
        return transformer

    def load_model(self, model_location: str) -> None:
        self._model = PCA.attach(
            model_location, sagemaker_session=self.executor.session
        )

    def upload_to_s3(self, data: DataFrame, name: str) -> str:
        # Idea: make this a mixin?
        LOGGER.info("Preparing input data for PCA")

        if not os.path.exists(self.local_save_folder):
            os.makedirs(self.local_save_folder)
        temp_location = f"{self.local_save_folder}/{name}.csv"

        LOGGER.info("Writing data to local machine")
        data.to_csv(temp_location, index=False, header=False)

        # Upload to S3
        s3_key = self.executor.upload_data(temp_location, prefix=self.input_data_prefix)
        LOGGER.info("Done with uploading data to S3")

        return s3_key

