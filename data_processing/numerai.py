# General python imports
from typing import List, Dict

# Data science imports
from pandas import DataFrame
import numpy as np

# Local imports
from .data_loader import DataLoader


class NumeraiDataLoader(DataLoader):

    index_column = "id"
    data_type_column = "data_type"
    time_column = "era"
    feature_columns = NotImplemented
    output_column = "target_kazutsugi"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_columns = self._get_feature_columns()

    @classmethod
    def _get_feature_columns(self) -> List[str]:
        """
        Defines all the feature columns for Numerai.
        """
        # NOTE: When making this general, maybe let this be overwritable?
        features = []
        column_ranges: List[Dict] = [
            {"prefix": "feature_intelligence", "range": range(1, 12 + 1)},
            {"prefix": "feature_charisma", "range": range(1, 86 + 1)},
            {"prefix": "feature_strength", "range": range(1, 38 + 1)},
            {"prefix": "feature_dexterity", "range": range(1, 14 + 1)},
            {"prefix": "feature_constitution", "range": range(1, 114 + 1)},
            {"prefix": "feature_wisdom", "range": range(1, 46 + 1)},
        ]
        for column in column_ranges:
            for index in column["range"]:
                features.append(f"{column['prefix']}{str(index)}")
        return features

    def format_predictions(
        self, Y_pred: DataFrame, all_data: bool = False
    ) -> DataFrame:
        """
        Formats the predictions by setting index and columns

        Arguments:
            Y_pred: dataframe with the predictions
            all_data: whether all data was used

        Returns:
            The formatted DataFrame
        """
        if all_data:
            Y_labels = self.data
        else:
            Y_labels = self.test_data
        # Format index and columns
        Y_pred = Y_pred.set_index(Y_labels.index, inplace=False)
        output_columns = [
            column.replace("target", "prediction") for column in [self.output_column]
        ]
        Y_pred = Y_pred.set_axis(output_columns, axis=1, inplace=False)
        return Y_pred

    def score_data(self, Y_pred: DataFrame, all_data: bool = False) -> float:
        """
        Scores the data versus the predictions.
        For numerai, corretation coefficient is used.

        Arguments:
            Y_pred: the predicted values
            all_data: Whether to use the complete dataset to compare to, or just the test set

        Returns:
            The scoring metric (correlation coefficient)
        """
        if all_data:
            Y_labels = self.data
        else:
            Y_labels = self.test_data
        Y_labels = Y_labels.loc[:, self.output_column]
        metric = self.score_correlation(Y_labels, Y_pred)
        return metric

    def score_correlation(self, labels: DataFrame, prediction: DataFrame) -> float:
        """
        Scores the correlation as defined by the Numerai tournament rules.

        Arguments:
            labels: The real labels of the output
            prediction: The predicted labels

        Returns:
            The correlation coefficient
        """
        ranked_prediction = prediction.rank(pct=True, method="first")
        return np.corrcoef(labels, ranked_prediction, rowvar=False)[0, 1]
