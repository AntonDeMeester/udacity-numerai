# Python imports
import logging
from typing import Iterable, Callable

# Data science imports
from pandas import DataFrame

# Local imports
from data_processing.data_loader import DataLoader

LOGGER = logging.getLogger(__name__)

class EasyCombiner:
    """
    This combines models in a dumb way to optimise multiple datasets.
    """


    def __init__(self, Y_labels: DataFrame, *Y_test: Iterable[DataFrame], score_function: Callable):
        self.Y_labels: DataFrame = Y_labels
        self.number_of_predictions = len(Y_test)
        self.Y_test: Iterable[DataFrame] = Y_test
        self.score_function = score_function


    def combine(self, steps_per_prediction: int = 10):
        """
        Combines a number of output predictions to provide a weighted output.
        This is a naivie combiner that assigns weights from 0 until and including step per prediction.
        """
        LOGGER.info("Starting to combine")

        # TODO Make this recursive
        total_number_of_steps = (steps_per_prediction + 1) ** (self.number_of_predictions)
        columns = self.Y_test[0].columns
        indexes = self.Y_test[0].index

        best_score: int = -1
        best_weights: Iterable[int] = []
        best_Y: DataFrame = DataFrame()

        for i in range(total_number_of_steps):
            weights = self._convert_number_to_weights(i, steps_per_prediction)
            Y_attempt = DataFrame(0, columns=columns, index=indexes)
            for j, test in enumerate(self.Y_test):
                Y_attempt += test * weights[j]
            score = self.score_function(self.Y_labels, Y_attempt)
            if score > best_score:
                best_score = score
                best_weights = weights
                best_Y = Y_attempt   
                LOGGER.info(f"Got a new best score for a combination: {best_score} with weights {best_weights}")

        LOGGER.info(f"Ended the combination job: score {best_score} with weights {best_weights}")
        return best_score, weights, best_Y 

    def _convert_number_to_weights(self, index, steps_per_prediction):
        weights = []
        new_index = index
        for i in range(self.number_of_predictions):
            new_weight = (new_index % (steps_per_prediction + 1)) / (steps_per_prediction + 1)
            weights.append(new_weight)
            new_index = new_index // (steps_per_prediction + 1)
        total_weight = sum(weight for weight in weights)
        if total_weight != 0:
            weights = [weight / total_weight for weight in weights]
        return weights

