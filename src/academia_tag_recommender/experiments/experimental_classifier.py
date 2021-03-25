import os
from pathlib import Path
from joblib import dump, load
import numpy as np
import time

from academia_tag_recommender.evaluator import Evaluator
from academia_tag_recommender.definitions import MODELS_PATH
from academia_tag_recommender.experiments.transformer import Transformer

PATH = Path(MODELS_PATH) / 'experimental_classifier'


def available_classifier_paths(*restrictions):
    paths = list(PATH.glob('*.joblib'))
    for restriction in restrictions:
        paths = [path for path in paths if restriction in path.name]
    return paths


class ExperimentalClassifier:

    def __init__(self, transformer, classifier, name):
        self.transformer_path = transformer.path
        self.classifier = classifier
        self.name = name
        self.path = ExperimentalClassifier._to_path(transformer, name)

    def __str__(self):
        return self.name

    def fit(self, X):
        transformer = load(self.transformer_path)
        return transformer.fit(X)

    def transform(self, X):
        transformer = load(self.transformer_path)
        return transformer.transform(X)

    def train(self, X, y):
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
        start = time.time()
        X = np.array(X)
        prediction = self.classifier.predict(X)
        end = time.time()
        self.test_time = end - start
        self.evaluate(y, prediction)
        dump(self, self.path)

    def evaluate(self, y, prediction):
        self.evaluation = Evaluator(y, prediction)

    def predict(self, X):
        return self.classifier.predict(X)

    @ classmethod
    def load(clf, transformer, classifier, name):
        path = clf._to_path(transformer, name)
        if os.path.isfile(path):
            return load(path)
        else:
            print('Classifier not available. Initiating data.')
            return clf(transformer, classifier, name)

    @ staticmethod
    def _to_path(transformer, classifier):
        return PATH / 'name={}&{}.joblib'.format(classifier, transformer)
