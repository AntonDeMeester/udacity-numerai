from data_processing.numerai import NumeraiDataLoader
from models.hunga_bunga import HungaBungaRegressor
from models.pytorch.aws_models import AwsTwoLayerLinearNeuralNetwork
from executors.sagemaker import Sagemaker
from sagemaker.pytorch.model import PyTorchPredictor

import logging

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
"""
data_loader = NumeraiDataLoader(data=data_loader.data.sample(frac=0.01))
model = HungaBungaRegressor(data=data_loader)
"""
model = AwsTwoLayerLinearNeuralNetwork(data=data_loader, aws_executor=sagemaker)
# model.train()
# model.load_estimator("sagemaker-pytorch-2019-09-08-11-24-24-412")
model._predictor = PyTorchPredictor("sagemaker-pytorch-2019-09-08-13-06-53-003", sagemaker.session)
Y_pred = model.predict()

print(data_loader.score_data(Y_pred))
