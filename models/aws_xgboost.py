# General python imports
from typing import Dict, Any

# Amazon imports
from sagemaker.tuner import IntegerParameter, ContinuousParameter

# Local imports
from .aws_base import AwsEstimator


class AwsXGBoost(AwsEstimator):
    """
    XG Boost implementation for AWS
    """

    container_name: str = "xgboost"
    name: str = "XGBoost"


class LinearAwsXGBooost(AwsXGBoost):
    default_hyperparameters: Dict[str, Any] = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weights": 6,
        "subsample": 0.8,
        "objective": "reg:linear",
        "early_stopping_rounds": 10,
        "num_round": 200,
    }
    default_hyperparameter_tuning: Dict[str, Any] = {
        "max_depth": IntegerParameter(3, 12),
        "eta": ContinuousParameter(0.05, 0.5),
        "gamma": ContinuousParameter(0, 10),
        "subsample": ContinuousParameter(0.5, 0.9),
        "num_round": IntegerParameter(50, 400),
    }
    name: str = "linear_xgboost"


class BinaryAwsXGBoost(AwsXGBoost):
    default_hyperparameters: Dict = {
        "objective": "reg:logistic",
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weights": 6,
        "subsample": 0.8,
        "early_stopping_rounds": 10,
        "num_round": 200,
    }
    name: str = "binary_xgboost"


class MulticlassAwsXGBoost(AwsXGBoost):
    default_hyperparameters: Dict = {
        "objective": "multi:softmax",
        "num_class": 5,
        "max_depth": 6,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weights": 6,
        "subsample": 0.8,
        "early_stopping_rounds": 10,
        "num_round": 200,
    }
    name: str = "multiclass_xgboost"
