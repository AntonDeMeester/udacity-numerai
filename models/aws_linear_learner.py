# General python imports
from typing import Dict, Any

# Data science imports
from pandas import DataFrame

# Amazon imports
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter

# Local imports
from .aws_base import AwsEstimator


class AwsLinearLearner(AwsEstimator):
    container_name: str = "linear-learner"
    name: str = "linear_learner"
    default_hyperparameter_tuning: Dict[str, Any] = {
        "learning_rate": ContinuousParameter(0.01, 0.2),
        "mini_batch_size": IntegerParameter(250, 5000),
        "use_bias": CategoricalParameter([True, False]),
    }
    default_tuning_job_config = {
        "max_jobs": 20,
        "max_parallel_jobs": 3,
        "objective_metric_name": 'validation:objective_loss',
        "objective_type": "Minimize"
    }

    def _load_results(self, file_name: str) -> DataFrame:
        """
        Extension of the results to remove the score dict
        Arguments and return value the same as superclass
        """
        initial_df = super()._load_results(file_name)
        for _, row in initial_df.iterrows():
            try:
                row[0] = row[0].replace('{"score":', "").replace("}", "")
            except IndexError:
                pass
        initial_df = initial_df.astype("float32")
        return initial_df

    
class LinearAwsLinearLearner(AwsLinearLearner):
    default_hyperparameters: Dict = {
        "predictor_type": "regressor",
        "feature_dim": 1,
        "epochs": 10,
        "loss": "auto",
        "optimizer": "auto",
    }
    name: str = "linear_learner"

class BinaryAwsLinearLearner(AwsEstimator):
    default_hyperparameters: Dict = {
        "predictor_type": "binary_classifier",
        "binary_classifier_model_selection_criteria": "cross_entropy_loss",
        "feature_dim": 1,
        "epochs": 10,
        "loss": "auto",
        "optimizer": "auto",
    }
    name: str = "binary_linear_learner"

class MulticlassAwsLinearLearner(AwsEstimator):
    default_hyperparameters: Dict = {
        "predictor_type": "multiclass_classifier",
        "num_classes": 5,
        "balance_multiclass_weights": True,
        "feature_dim": 1,
        "epochs": 10,
        "loss": "auto",
        "optimizer": "auto",
    }
    name: str = "multiclass_linear_learner"
