# Python imports
import logging
import os

# Local imports
from data_processing.numerai import NumeraiDataLoader
from executors.numerai import Numerai
from executors.sagemaker import Sagemaker
from models.aws_linear_learner import LinearAwsLinearLearner
from models.aws_xgboost import LinearAwsXGBooost
from models.combiner import NaiveCombiner
from models.meta_model import MetaModel
from models.pytorch.aws_models import AwsTwoLayerLinearNeuralNetwork


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

numerai = Numerai()
data_location = numerai.download_latest_data("data/temp")
sagemaker = Sagemaker()
data_loader = NumeraiDataLoader(
    local_data_location=os.path.join(data_location, "numerai_training_data.csv")
)

# Loading models and Predictors
xgboost = LinearAwsXGBooost(data=data_loader, aws_executor=sagemaker)
xgboost.load_model("xgboost-190911-2045-001-9ea55a3c")
linear_learner = LinearAwsLinearLearner(data=data_loader, aws_executor=sagemaker)
linear_learner.load_model("linear-learner-190911-2051-010-c1988378")
tl_nn = AwsTwoLayerLinearNeuralNetwork(data=data_loader, aws_executor=sagemaker)
tl_nn.load_estimator("sagemaker-pytorch-2019-09-11-19-47-58-170")

# Creating the meta model to combine them
combiner = NaiveCombiner(data_loader.execute_scoring, 10)
meta_model = MetaModel(
    data=data_loader, models=[xgboost, linear_learner, tl_nn], combiner=combiner
)

# We don't train, we set the weights to what we know
meta_model.model_weights = [0.39999999999999997, 0.14999999999999997, 0.45]

# Predict production data
production_data = NumeraiDataLoader(
    local_data_location=os.path.join(data_location, "numerai_tournament_data.csv")
)
predictions = meta_model.predict(data_loader=production_data, all_data=True)

predictions = production_data.format_predictions(predictions, all_data=True)
numerai.upload_predictions(predictions, local_folder="data/temp/predictions")
