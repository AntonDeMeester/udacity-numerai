# Local imports
from models.aws_pytorch import AwsPytorch


class AwsTwoLayerLinearNeuralNetwork(AwsPytorch):
    """
    Model to be used for AWS training.
    """

    default_hyperparameters = {
        "D_in": NotImplemented,  # Will be loaded based on data input
        "Hidden": 150,
        "D_out": 1,
        "epochs": 10,
        "lr": 0.001,
    }

    train_entry_point: str = "two_layer_linear.py"
    predict_entry_point: str = "two_layer_linear.py"
    source_directory: str = "models/pytorch"
    name: str = "pytorch_two_linear"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_hyperparameters["D_in"] = len(self.data.feature_columns)
