"""This module handles classifier calculation."""
from academia_tag_recommender.evaluator import Evaluator
from academia_tag_recommender.definitions import MODELS_PATH
from academia_tag_recommender.test_train_data import fit_labels
from joblib import dump, load
from pathlib import Path
import numpy as np
import time

DATA_FOLDER = Path(MODELS_PATH) / 'classifier' / 'multi-label'


class Classifier:
    """This classifier models holds the actual classifier, along with evaluation statistics.
    """

    def __init__(self, classifier, preprocessing):
        self.classifier = classifier
        self.preprocessing = preprocessing

    def fit(self, X, y):
        """Fit classifier to given data and track training time.

        :param X: The X data
        :param y: The y data
        """
        start = time.time()
        self.classifier.fit(X, y)
        end = time.time()
        self.training_time = end - start

    def score(self, X, y):
        """Score the correctness of the prediction for the given data.

        :param X: The X data
        :param y: The y data
        """
        prediction = self.classifier.predict(X)
        self.evaluate(y, prediction)

    def evaluate(self, y, prediction):
        """Evaluate the similarity between y and prediction.

        :param y: The true label data
        :param prediction: The predicted label data
        """
        self.evaluation = Evaluator(y, prediction)

    def file_name(self):
        """Create a filename (.joblib) using the classifier and preprocessing information"""
        return 'name={}&{}.joblib'.format(self.classifier, self.preprocessing)

    def save(self):
        """Save a copy of self"""
        path = DATA_FOLDER / self.file_name()
        dump(self, path)
        return path

    def __str__(self):
        return str(self.classifier)
