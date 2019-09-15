import logging
import os
from typing import List

import pandas as pd

from data_processing.numerai import NumeraiDataLoader
from models.base import BaseModel
from models.aws_linear_learner import LinearAwsLinearLearner
from models.aws_xgboost import LinearAwsXGBooost, LinearAwsXGBooost
from executors.sagemaker import Sagemaker
from executors.numerai import Numerai
from models.easy_combiner import EasyCombiner

from default_config import DEFAULT_MODELS


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

sagemaker = Sagemaker()
data_loader = NumeraiDataLoader(local_data_location="data/numerai_training_data.csv")
data_loader.add_to_cache("local", "train", "data/temp/linear_learner/train.csv")
data_loader.add_to_cache(
    "local", "validation", "data/temp/linear_learner/validation.csv"
)
data_loader.add_to_cache("local", "test", "data/temp/linear_learner/test.csv")
data_loader.add_to_cache(
    "s3", "train", "s3://sagemaker-eu-west-1-729071960169/data/input_data/train.csv"
)
data_loader.add_to_cache(
    "s3",
    "validation",
    "s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/validation.csv",
)
data_loader.add_to_cache(
    "s3",
    "test",
    "s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/test.csv",
)


linear_learner = LinearAwsLinearLearner(data=data_loader, aws_executor=sagemaker)
# linear_learner.tune()
linear_learner.load_model("linear-learner-190831-1633-005-9e7fae2a")
# Y_test_ll = linear_learner.batch_predict()
Y_test_ll = linear_learner._load_results("test")

xgboost = LinearAwsXGBooost(data=data_loader, aws_executor=sagemaker)
# xgboost.train()
# xgboost.tune()
xgboost.load_model("xgboost-190831-1837-001-c34eb1a8")
# Y_test_xg = xgboost.batch_predict()
Y_test_xg = xgboost._load_results("test")

score_ll = data_loader.score_data(Y_test_ll)
score_xgb = data_loader.score_data(Y_test_xg)
Y_labels = data_loader.test_data.loc[:, data_loader.output_column]

combiner = EasyCombiner(
    Y_labels, Y_test_xg, Y_test_ll, score_function=data_loader.score_correlation
)
score, weights, y_predict = combiner.combine(10)
y_predict.to_csv("data/temp/predictions/combination.csv")

"""
Y_test_ll = data_loader.format_predictions(Y_test_ll)
Y_test_xgb = data_loader.format_predictions(Y_test_xgb)

print(f"The linear learner scored {score_ll}.")
print(f"The xgboost model scored {score_xgb}.")

"""
# Predict production data
production_data = NumeraiDataLoader(
    local_data_location="data/numerai/numerai_tournament_data.csv"
)
# Y_pred_prod_xg = xgboost.batch_predict(data_loader=production_data, all_data=True, name="predictions_xg")
Y_pred_prod_xg = xgboost._load_results("predictions_xg")
# Y_pred_prod_ll = linear_learner.batch_predict(data_loader=production_data, all_data=True, name="predictions_ll")
Y_pred_prod_ll = linear_learner._load_results("predictions_ll")
Y_total = Y_pred_prod_xg * weights[0] + Y_pred_prod_ll * weights[1]


"""
model_instances: List[BaseModel] = []
d
for model in DEFAULT_MODELS:
    model_instance = model(data=data_loader, aws_executor=sagemaker)
    model_instance.train()
    Y_pred = model_instance.batch_predict(all_data=False)
    data_loader.add_to_cache("predictions", model_instance.name, Y_pred)
"""

numerai = Numerai()
Y_total = production_data.format_predictions(Y_total, all_data=True)
numerai.upload_predictions(Y_total, local_folder="data/temp/predictions")
