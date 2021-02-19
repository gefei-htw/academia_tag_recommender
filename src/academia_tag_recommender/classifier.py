"""This module handles classifier calculation."""
from academia_tag_recommender.classwise_classifier import ClasswiseClassifier
from academia_tag_recommender.evaluator import Evaluator
from academia_tag_recommender.definitions import MODELS_PATH
from joblib import dump
from pathlib import Path
import time

DATA_FOLDER = Path(MODELS_PATH) / 'classifier' / 'multi-label'


def _path(subfolder=False):
    return DATA_FOLDER / subfolder if subfolder else DATA_FOLDER


def available_classifier_paths(subfolder=False, recursive=False):
    if recursive:
        return list(_path(subfolder).glob('**/*.joblib'))
    else:
        return list(_path(subfolder).glob('*.joblib'))


class Classifier:
    """This classifier models holds the actual classifier, along with evaluation statistics.
    """

    def __init__(self, classifier, preprocessing, name_prefix=False):
        if isinstance(classifier, ClasswiseClassifier):
            classifier.set_name(name_prefix)
        self.classifier = classifier
        self.preprocessing = preprocessing
        self.name_prefix = name_prefix

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
        start = time.time()
        self.prediction = self.predict(X)
        end = time.time()
        self.test_time = end - start
        self.evaluate(y, self.prediction)

    def predict(self, X):
        return self.classifier.predict(X)

    def evaluate(self, y, prediction):
        """Evaluate the similarity between y and prediction.

        :param y: The true label data
        :param prediction: The predicted label data
        """
        self.evaluation = Evaluator(y, prediction)

    def file_name(self):
        """Create a filename (.joblib) using the classifier and preprocessing information"""
        return 'name={}&{}.joblib'.format(str(self), self.preprocessing)

    def save(self, subfolder=False):
        """Save a copy of self"""
        path = _path(subfolder) / self.file_name()
        dump(self, path)
        return path

    def __str__(self):
        return self.name_prefix if hasattr(self, 'name_prefix') and self.name_prefix else str(self.classifier).replace(' ', '').replace('\n', '')
