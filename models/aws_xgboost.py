# General python imports
from typing import Dict

# Local imports
from .aws_base import AwsEstimator


class AwsXGBoost(AwsEstimator):
    """
    XG Boost implementation for AWS
    """

    default_hyperparameters: Dict = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weights": 6,
        "subsample": 0.8,
        "objective": "reg:linear",
        "early_stopping_rounds": 10,
        "num_round": 200,
    }
    container_name: str = "xgboost"
    name: str = "XGBoost"
