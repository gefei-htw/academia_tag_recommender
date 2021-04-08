import os
from pathlib import Path
from joblib import dump, load
import numpy as np
import time

from academia_tag_recommender.evaluator import Evaluator
from academia_tag_recommender.definitions import MODELS_PATH

PATH = Path(MODELS_PATH) / 'experimental_classifier'


def available_classifier_paths(*restrictions):
    """Returns all classifier paths with given restrictions.

    Args:
        restrictions: Strings that should be included in the paths as :class:`str`.

    Returns:
        The path of the classifiers as :class:`list` of :class:`pathlib.Path`.
    """
    paths = list(PATH.glob('*.joblib'))
    for restriction in restrictions:
        paths = [path for path in paths if restriction in path.name]
    return paths


class ExperimentalClassifier:
    """Wrapper for classifiers used for experiments.

    Attributes:
        name: The experimental classifiers name as :class:`str`.
        path: The experimental classifiers path on the disc as :class:`pathlib.Path`.
        transformer_path: The transformers path on the disc as :class:`pathlib.Path`.
        classifier: The actual classifier.
        training_time: The time passed while training in seconds as :class:`float`.
        test_time: The time passed while testing in seconds as :class:`float`.
        evaluation: The evaluation metrics for the prediction as :class:`academia_tag_recommender.evaluator.Evaluator`.
    """

    def __init__(self, transformer, classifier, name):
        self.transformer_path = transformer.path
        self.classifier = classifier
        self.name = name
        self.path = ExperimentalClassifier._to_path(transformer, name)

    def __str__(self):
        """String representation."""
        return self.name

    def fit(self, X):
        """Fit transformer using X.

        Args;
            X: The original samples as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        transformer = load(self.transformer_path)
        return transformer.fit(X)

    def transform(self, X):
        """Transform X.

        Args;
            X: The original samples as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        transformer = load(self.transformer_path)
        return transformer.transform(X)

    def train(self, X, y):
        """Train classifier using X and save on disk.

        Args;
            X: The original samples as :class:`list`.
            y: The one hot encoded labels of the samples as :class:`list`.
        """
        if os.path.isfile(self.path):
            pass
        else:
            start = time.time()
            X = np.array(X)
            self.classifier.fit(X, y)
            end = time.time()
            self.training_time = end - start
            dump(self, self.path)

    def score(self, X, y):
        """Predict labels for X and compare to y.

        Args;
            X: The original samples as :class:`list`.
            y: The one hot encoded labels of the samples as :class:`list`.
        """
        start = time.time()
        X = np.array(X)
        prediction = self.classifier.predict(X)
        end = time.time()
        self.test_time = end - start
        self.evaluate(y, prediction)
        dump(self, self.path)

    def evaluate(self, y, prediction):
        """Calculate evaluation metrics.

        Args;
            y: The original one hot encoded labels of the samples as :class:`list`.
            prediction: The predicted one hot encoded labels of the samples as :class:`list`.
        """
        self.evaluation = Evaluator(y, prediction)

    def predict(self, X):
        """Predict label data for X.

        Args;
            X: The original samples as :class:`list`.

        Returns:
            The predicted one hot encoded labels of the samples as :class:`list`.
        """
        return self.classifier.predict(X)

    @ classmethod
    def load(clf, transformer, classifier, name):
        """Loads an existing ExperimentalClassifier instance from disc or creates new if none exists.

        Args:
            transformer: The transformers name on the disc as :class:`academia_tag_recommender.experiments.Transformer`.
            classifier: The actual classifier.
            name: The experimental classifiers name as :class:`str`.

        Returns
            The configured :class:`ExperimentalClassifier` instance.
        """
        path = clf._to_path(transformer, name)
        if os.path.isfile(path):
            return load(path)
        else:
            print('Classifier not available. Initiating data.')
            return clf(transformer, classifier, name)

    @ staticmethod
    def _to_path(transformer, classifier):
        """Converts transformer and classifier into path name.

        Args:
            transformer: The transformers name on the disc as :class:`Transformer`.
            classifier: The actual classifier.

        Returns:
            The Path of the classifier as :class:`pathlib.Path`.
        """
        return PATH / 'name={}&{}.joblib'.format(classifier, transformer)
