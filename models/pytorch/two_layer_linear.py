# General python imports
import argparse
from io import StringIO
import json
import os
from six import BytesIO
import sys

# Data science imports
import numpy as np
import pandas as pd

# pytorch
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils import data


# NOTE: inspired heavily by https://github.com/udacity/ML_SageMaker_Studies/blob/master/Moon_Data/source_solution/train.py
CONTENT_TYPE = "application/x-npy"


class TwoLayerLinearNeuralNetwork(nn.Module):
    def __init__(self, D_in: int, Hidden: int, D_out: int):
        """
        Initialises the Neural network

        Arguments:
            D_in: the input dimension
            Hidden; the hidden dimension
            D_out: The output dimension
        """
        super().__init__()
        self.input_linear = nn.Linear(D_in, Hidden)
        self.output_linear = nn.Linear(Hidden, D_out)

    def forward(self, x):
        out = functional.relu(self.input_linear(x))
        out = self.output_linear(out)
        return out


def model_fn(model_dir):
    """
    Loads the model from the model directory.
    """
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, "model_info.pth")
    with open(model_info_path, "rb") as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLayerLinearNeuralNetwork(
        model_info["D_in"], model_info["Hidden"], model_info["D_out"]
    )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f))

    return model.to(device)


# Load the training data from a csv file
def _get_train_loader(batch_size, data_dir):
    print("Get data loader.")

    # read in csv file
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None)

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float()
    # features are the rest
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save trained model, after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)


def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, "model_info.pth")
    with open(model_info_path, "wb") as f:
        model_info = {"D_in": args.D_in, "Hidden": args.Hidden, "D_out": args.D_out}
        torch.save(model_info, f)


# Prediction functions
def input_fn(serialized_input_data, content_type):
    print("Deserializing the input data.")
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream, allow_pickle=True)
    raise Exception(
        "Requested unsupported ContentType in content_type: " + content_type
    )


def output_fn(prediction_output, accept):
    print("Serializing the generated output.")
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)


def predict_fn(input_data, model):
    print("Predicting class labels for the input data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(input_data.astype("float32"))
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()

    return result


# Main loading function for train
if __name__ == "__main__":
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    # Training Parameters, given
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    # Model parameters
    parser.add_argument(
        "--D_in",
        type=int,
        default=310,
        metavar="IN",
        help="number of input features to model (default: 310)",
    )
    parser.add_argument(
        "--Hidden",
        type=int,
        default=100,
        metavar="H",
        help="hidden dim of model (default: 100)",
    )
    parser.add_argument(
        "--D_out",
        type=int,
        default=1,
        metavar="OUT",
        help="output dim of model (default: 1)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # get train loader
    train_loader = _get_train_loader(
        args.batch_size, args.data_dir
    )  # data_dir from above..

    # Initialise model and put to device
    model = TwoLayerLinearNeuralNetwork(
        D_in=args.D_in, Hidden=args.Hidden, D_out=args.D_out
    ).to(device)

    # Given: save the parameters used to construct the model
    save_model_params(model, args.model_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, args.epochs, optimizer, criterion, device)
