from data_processing.data_loader import DataLoader
from models.linear_learner import AwsLinearLearner
from exectors.sagemaker import Sagemaker

sagemaker = Sagemaker()
data_loader = DataLoader(
    local_data_location="data/numerai_training_data.csv"
)
linear_learner = AwsLinearLearner(
    data=data_loader,
    aws_executor=sagemaker,
)
linear_learner.train()