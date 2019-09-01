

from models.aws_linear_learner import LinearAwsLinearLearner, MulticlassAwsLinearLearner
from models.aws_xgboost import LinearAwsXGBooost, MulticlassAwsXGBoost

# Linear Linear learrner
linear_learner = LinearAwsLinearLearner
multiclass_linear = MulticlassAwsLinearLearner
linear_xgboost = LinearAwsXGBooost
multiclass_xgboost = MulticlassAwsXGBoost

# Set hyperparameters
# Or in dict?



DEFAULT_MODELS = [
    linear_learner,
    multiclass_linear,
    linear_xgboost,
    multiclass_xgboost,
]