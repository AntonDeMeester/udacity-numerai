# Python imports
from abc import ABC
import logging
from typing import Iterable, Callable, List, Collection

# Data science imports
from pandas import DataFrame

# Local imports
from data_processing.data_loader import DataLoader

LOGGER = logging.getLogger(__name__)


class Combiner(ABC):
    """
    A base class for any Combiners
    """

    def __init__(self, score_function: Callable[[DataFrame, DataFrame], float]):
        self.score_function = score_function

    def combine(
        self, lables: DataFrame, predictions: Collection[DataFrame]
    ) -> List[float]:
        """
        Combines predictions to provide a better aggregate prediction.

        Arguments:
            * labels: The correct data
            * predictions: The list of predictions by the models

        Returns:
            * A list of float with the individual weights. Sum of the weights will be 1
        """
        return NotImplemented


class NaiveCombiner(Combiner):
    """
    This combines models in a dumb way to optimise multiple datasets.
    """

    def __init__(
        self,
        score_function: Callable[[DataFrame, DataFrame], float],
        number_of_steps: int = 10,
    ):
        """
        Initialises the Naive Combiner. 
        Arguments:
            * All arguments for the BaseCombiner
            * number_of_steps: The number of steps to use. Default 10.
        """
        assert number_of_steps >= 1, "Step size must be at least 1"
        super().__init__(score_function)

        self.number_of_steps = number_of_steps

    def combine(
        self, lables: DataFrame, predictions: Collection[DataFrame]
    ) -> List[float]:
        """
        Combines a number of output predictions to provide a weighted output.
        This is a naivie combiner that assigns weights from 0 until and including step per prediction.

        Arguments:
            * labels: The correct data
            * predictions: The list of predictions by the models

        Returns:
            * A list of float with the individual weights. Sum of the weights will be 1
        """
        LOGGER.info("Starting to combine")

        number_of_predictions = len(predictions)
        total_number_of_steps = (self.number_of_steps + 1) ** (number_of_predictions)
        columns = lables.columns
        indexes = lables.index

        best_score: float = -1
        best_weights: List[float] = []
        best_Y: DataFrame = None

        for i in range(total_number_of_steps):
            weights = self._convert_number_to_weights(i, self.number_of_steps)
            Y_attempt = DataFrame(0, columns=columns, index=indexes)
            for j, test in enumerate(predictions):
                Y_attempt += test * weights[j]
            score = self.score_function(lables, Y_attempt)
            if score > best_score:
                best_score = score
                best_weights = weights
                best_Y = Y_attempt
                LOGGER.info(
                    f"Got a new best score for a combination: {best_score} with weights {best_weights}"
                )

        LOGGER.info(
            f"Ended the combination job: score {best_score} with weights {best_weights}"
        )
        return best_weights

    def _convert_number_to_weights(self, index, steps_per_prediction):
        weights = []
        new_index = index
        for i in range(self.number_of_predictions):
            new_weight = (new_index % (steps_per_prediction + 1)) / (
                steps_per_prediction + 1
            )
            weights.append(new_weight)
            new_index = new_index // (steps_per_prediction + 1)
        total_weight = sum(weight for weight in weights)
        if total_weight != 0:
            weights = [weight / total_weight for weight in weights]
        return weights
