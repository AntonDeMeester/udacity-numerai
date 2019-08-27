# General python imports
from abc import ABC
import os
from typing import Optional, Dict, Any

# Data science imports
from pandas import DataFrame

# Other imports
import numerapi


class Numerai:
    """
    Executor class to deal with Numerai uploads and downloads

    Main important methods are format_predictions, download_latest_data and upload_predictions
    """

    def __init__(
        self, public_id: Optional[str] = None, secret_key: str = Optional[None]
    ):
        """
        Initializes the Numerai class to execute numerai things.
        Can take the public id and secret key, else will load them from the env variables
        """
        if public_id is None:
            public_id = os.environ.get("NUMERAI_PUBLIC_ID", None)
        if secret_key is None:
            secret_key = os.environ.get("NUMERAI_SECRET_KEY", None)
        assert public_id is not None and secret_key is not None, (
            "You need to either provide the numerai public and and secret key, "
            + "or you need to put them in environment variables."
        )

        self.napi: numerapi.NumerAPI = numerapi.NumerAPI(
            public_id=public_id, secret_key=secret_key
        )

    def format_predictions(
        self, predictions: DataFrame, local_folder: str, name: str = "predictions"
    ) -> str:
        """
        Formats the predictions and saves them to a local file.

        Arguments:
            predictions: The dataframe with the predictions, with index and all output columns
            local_folder: The folder to save thepredictions to
            name: The name of the local file

        Returns:
            local_file_location: The location of the file locally.
        """
        local_file_location = os.path.join(local_folder, f"{name}.csv")
        predictions.to_csv(local_file_location, index=True, header=True)
        return local_file_location

    def download_latest_data(self, local_folder: str, name: str = None) -> str:
        """
        Downloads the latest data for the tournament        

        Arguments:
            predictions: The dataframe with the predictions, with index and all output columns
            local_folder: The folder to save thepredictions to
        """
        local_file_folder = self.napi.download_current_dataset(
            dest_path=local_folder, dest_filename=name, upzip=True
        )
        return local_file_folder

    def upload_predictions(
        self,
        predictions: DataFrame,
        local_folder: str = "data/temp",
        name: str = "predictions",
    ) -> bool:
        """
        Formats and uploads the predictions.

        Arguments:
            predictions: The dataframe with the predictions, with index and all output columns
            local_folder: The folder to save thepredictions to
            name: The name of the local file

        Returns:
            success: Whether the upload was successful
        """
        local_file = self.format_predictions(predictions, local_folder, name)
        return self.upload_predictions_csv(local_file)

    def upload_predictions_csv(self, file_location: str) -> bool:
        """
        Uploads the predictions to Numerai

        Arguments:
            file_location: the location of the file in the system

        Returns:
            success: Whether the upload was successful
        """
        submission_id = self.napi.upload_predictions(file_location)
        success = self.napi.check_submission_successful(submission_id)
        return success

    def stake(self, amount: float, confidence: float) -> None:
        """
        Stakes with the current predictions
        """
        self.napi.stake(confidence, amount)
