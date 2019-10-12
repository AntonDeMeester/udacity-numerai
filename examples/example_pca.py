# Python imports
import logging
import os

# Local imports
from data_processing.numerai import NumeraiDataLoader
from data_processing.aws_pca import AwsPCA
from executors.numerai import Numerai
from executors.sagemaker import Sagemaker
from models.aws_linear_learner import LinearAwsLinearLearner
from models.aws_xgboost import LinearAwsXGBooost
from models.combiner import NaiveCombiner
from models.meta_model import MetaModel
from models.pytorch.aws_models import AwsTwoLayerLinearNeuralNetwork


def example_pca():
    data_location = "data/temp/numerai_dataset_180"
    sagemaker = Sagemaker()
    data_loader = NumeraiDataLoader(
        local_data_location=os.path.join(data_location, "numerai_training_data.csv")
    )
    data_loader._data = data_loader.data.sample(frac=0.1)
    pca = AwsPCA(
        data=data_loader,
        aws_executor=sagemaker,
        columns="__all__",
        number_of_components=0.3,
    )
    pca.load_model("pca-2019-10-12-15-08-18-889")
    output_data = pca.process_data()
    output_data.head()

    print(output_data.columns)
