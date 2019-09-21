# AInterface
Want to combine multiple models together to make a super model? Getting stuck on the fact that they have different in- and output methods?

No more, with **AInterface** the interface to combine the latest and greatest of machine learning algorithm to make an even latester and greatester algorithm.

Using 3 algorithms just in 10 lines of code.

```python
# Data loading
data_loader = NumeraiDataLoader(local_data_location="data.csv")

# Initial models
sagemaker = Sagemaker()
xgboost = LinearAwsXGBooost(data=data_loader, aws_executor=sagemaker)
linear_learner = LinearAwsLinearLearner(data=data_loader, aws_executor=sagemaker)
tl_nn = AwsTwoLayerLinearNeuralNetwork(data=data_loader, aws_executor=sagemaker)

# How to combine the models
combiner = NaiveCombiner(data_loader.execute_scoring, 10)
meta_model = MetaModel(
    data=data_loader, models=[xgboost, linear_learner, tl_nn], combiner=combiner
)

# Train and predict
meta_model.train()
predictions = meta_model.predict()
```

Examples of the entire flow can be found in the examples folder, both for fully training as only loading.

## Setup

Install the requirements with `pip install` or with `pipenv install`.

## Usage

Using AInterface for your own project needs 3 steps:

1. Define your own `DataLoader`
2. Define your own (`Meta`)`Models`
3. If using a `MetaModel`, define a `Combiner`

And then you're set up to training and predicting with `model.train()` and `model.predict()`.

### DataLoaders

A `DataLoader` handles the loading of the data and providing a uniform interface to the models. It automatically splits the data in train, validation and test data sets. It can split the data into different data sets to train different models with a subset of the data. And lastly, it provides some caching to improve performance.

To define your own `DataLoader` for your own data set, subclass the `data_processing.data_loader.DataLoader` class, and set the `NotImplemented` class variables:

1. `feature_columns`
2. `output_column`

These will be used to select the feature to train on in your data, and the output column.

You can also override the following parameters if necessary, to provide better loading of the data:

1. `index_column`
2. `data_type_column` (Not yet used)
3. `time_column` (Not yet used)

### Models

#### Initializing

A general `Model` requires one input argument for initialization, a `DataLoader` instance. Other models can extend this, for example with a sagemaker instance.

#### Methods

The model is the uniform interface for any machine learning model. It contains three main functions:

1. `train`
2. `tune` (defaulted to train if not implemented)
3. `predict`
4. `load_model`

The `train` and `tune` functions will work without any input, but can contain arguments such as hyperparameters.

The `predict` function takes two optional arguments: a `DataLoader` and a `boolean` whether to just predict the test data or all data. If nothing is provided, the `predict` function will predict the test data of the current data loaded in the model.

`load_model` loads an already existing model so it can be used directly for predictions.

#### Creating your own model

To create your own model, you need to subclass `models.base.BaseMode` and implement the initialization (if you need any extra parameters), and at least the `train` and the `execute_prediction` function. For completeness sake, you can also implement `tune` and `load_model`.

The `train` function must be able to take no input, and must not return any data. It should train the model with the train and validation data.

The `execute_prediction` function take a pandas `DataFrame` input with the necessary features and returns a `DataFrame` with the predictions.

`tune` has the same requirements as `train` and `load_model` should accept any information it needs to load an existing model, should not return anything, and then the user should be able to predict without training.

### MetaModels and Combiners

#### MetaModel

A `MetaModel` is a way to combine different models to have an even better model. It requires a list of models as input which it can train, and a combination function. The combination function will define how to models need to work together to provide an even better result.

A `MetaModel` is also a model, so it can also be trained and used for predictions.

#### Combiner

A `Combiner` has one main function `combine` which accepts a list of predictions and the correct outputs, and returns a list of weights to use. The model can then use these weights to use later to provide unknown predictions.

## Provided information

### Data Processing

####  Numerai 

One `DataLoader` is provided, for the online data competition of [Numerai](https://numer.ai). It implements the output feature and the different feature columns.

### Models

#### AWS

Different Amazon models are implemented.

1. Two [Amazon Estimators](https://sagemaker.readthedocs.io/en/stable/estimators.html):
   1. XG Boost (`models.aws_xgboost`)
   2. Linear Learner (`models.aws_linear_learner`)
2. One [PyTorch model for AWS](https://sagemaker.readthedocs.io/en/stable/sagemaker.pytorch.html):
   1. Two layer linear network 
      * `models.pytorch.aws_models.AwsTwoLayerLinearNeuralNetwork` for the extension of `BaseModel`
      * `models.pytorch.two_layer_linear` for the AWS implementation


You can easily create your own AWS models by subclassing `AwsBase`, `AwsEstimator` or `AwsPyTorch`. They provide some basic help such as a sagemaker instance, basic data downloading, uploading, moving to S3 etc.


#### Hunga Bunga / SKLearn (Beta)

The [Hunga Bunga](https://github.com/ypeleg/HungaBunga) library provides the lazy man's implementation of all SKLearn's models. It goes over each default SKlearn models and select the best performing one. There is a Classifier and Regressor version.

NOTE: Because my computer gets stuck while executing these models, these have been **untested**.

### Executors

#### Sagemaker

A `Sagemaker` implementation is also there, that can get data from S3, and has some default parameters for model loading and creation.

#### Numerai

Numerai also has an API to download the latest tournament data and upload your predictions. A basic interface is also present to format the data correctly as required by Numerai.

# Feature requests

1. Implementation of Classifications
2. Implementation of other service providers
   1. Google Cloud
   2. Azure Cloud
3. Implementation of more model types
   1. Tensorflow
   2. Keras
   3. Docker images
4. Data pre-processing
   1. PCA
5. Refactoring of the PyTorch models
6. More combiners
7. More implemented models
8. Default configurations
9.  States
    1.  Saving to state
    2.  Loading from state
    3.  State validation
10. Possible refactor of hyperparameters etc to model initialisation
11. Error handling
12. Parallel training/prediction
    1.  For non-local models (i.e. starting all loading and predicting at the same time)
    2.  For local models