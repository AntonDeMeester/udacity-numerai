import logging

from data_processing.data_loader import DataLoader
from models.aws_linear_learner import AwsLinearLearner
from models.aws_xgboost import AwsXGBoost
from executors.sagemaker import Sagemaker

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

sagemaker = Sagemaker()
data_loader = DataLoader(local_data_location="data/numerai_training_data.csv")
data_loader.add_to_cache("local", "train", "data/temp/linear_learner/train.csv")
data_loader.add_to_cache("local", "validation", "data/temp/linear_learner/validation.csv")
data_loader.add_to_cache("local", "test", "data/temp/linear_learner/test.csv")
data_loader.add_to_cache("s3", "train", "s3://sagemaker-eu-west-1-729071960169/data/input_data/train.csv")
data_loader.add_to_cache("s3", "validation", "s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/validation.csv")
data_loader.add_to_cache("s3", "test", "s3://sagemaker-eu-west-1-729071960169/data/linear_learner/input_data/test.csv")

linear_learner = AwsLinearLearner(data=data_loader, aws_executor=sagemaker)
# linear_learner.train()
# linear_learner.load_model('linear-learner-2019-08-25-10-23-16-401')
# Y_test_ll = linear_learner.batch_predict()
Y_test_ll = linear_learner._load_results("test") # test

xgboost = AwsXGBoost(data=data_loader, aws_executor=sagemaker)
# xgboost.train()
# xgboost.load_model('xgboost-2019-08-25-10-48-40-209')
# Y_test_xg = xgboost.batch_predict()
Y_test_xgb = xgboost._load_results("test") # test

score_ll = data_loader.score_data(Y_test_ll)
score_xgb = data_loader.score_data(Y_test_xgb)

print(f"The linear learner scored {score_ll}.")
print(f"The xgboost model scored {score_xgb}.")
