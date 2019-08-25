# General python imports
from typing import Dict

# Data science imports
from pandas import DataFrame

# Local imports
from .aws_base import AwsEstimator

class AwsLinearLearner(AwsEstimator):
    default_hyperparameters: Dict = {
        "predictor_type": "regressor",
        "feature_dim": 1,
        "epochs": 10,
        "loss": "auto"
    }
    container_name: str = "linear-learner"
    name: str = "linear_learner"

    def _load_results(self, file_name: str) -> DataFrame:
        """
        Extension of the results to remove the score dict
        Arguments and return value the same as superclass
        """
        initial_df = super()._load_results(file_name)
        for _, row in initial_df.iterrows():
            try:
                row[0] = row[0].replace('{"score":', '').replace("}", "")
            except IndexError:
                pass
        initial_df = initial_df.astype('float32')
        return initial_df
