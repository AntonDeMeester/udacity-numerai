import logging

from data_processing.data_loader import DataLoader
from models.linear_learner import AwsLinearLearner
from executors.sagemaker import Sagemaker

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

sagemaker = Sagemaker()
data_loader = DataLoader(local_data_location="data/numerai_training_data.csv")
data_loader.add_to_cache("local", "train", "data/temp/linear_learner/train.csv")
data_loader.add_to_cache("local", "validation", "data/temp/linear_learner/test.csv")
data_loader.add_to_cache("s3", "train", "s3://sagemaker-eu-west-1-729071960169/data/input_data/train.csv")
data_loader.add_to_cache("s3", "validation", "s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/validation.csv")
linear_learner = AwsLinearLearner(data=data_loader, aws_executor=sagemaker)
# linear_learner.train()
# linear_learner.load_model('linear-learner-2019-08-22-19-24-55-421')
Y_test = linear_learner.batch_predict()
print(Y_test.head())