from pathlib import Path
from joblib import load, dump
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from academia_tag_recommender.data import documents
from academia_tag_recommender.definitions import MODELS_PATH

PATH = Path(MODELS_PATH) / 'experimental_data.joblib'
MIN_OCCURENCE = 70


class ExperimentalData:

    def __init__(self):
        X = self._get_X()
        y = self._get_y()
        y = self._get_y_reduced(y)
        self.X_train, self.X_test, self.y_train, self.y_test = self._get_test_train_data(
            X, y)

    def get_train_test_set(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _get_X(self):
        return np.array([document.text for document in documents])

    def _get_y(self):
        tags = [document.tags for document in documents]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(tags)
        self.label = mlb.classes_
        return np.array(y)

    def _get_test_train_data(self, X, y):
        X_train, y_train, X_test, y_test = self._iterative_train_test_split(
            X, y)
        return X_train, X_test, y_train, y_test

    def _iterative_train_test_split(self, X, y):
        stratifier = IterativeStratification(
            n_splits=2, order=2, sample_distribution_per_fold=[0.25, 0.75])
        train_indexes, test_indexes = next(stratifier.split(X, y))
        return X[train_indexes], y[train_indexes, :],  X[test_indexes], y[test_indexes, :]

    def _get_y_reduced(self, y):
        label_indices = self._get_indices_of_label_with_enough_occurence(y)
        self.label = self.label[label_indices]
        return np.array([[column for i, column in enumerate(row) if i in label_indices] for row in y])

    def _get_indices_of_label_with_enough_occurence(self, y):
        return [i for i, _ in enumerate(self.label) if np.sum(y[:, i]) >= MIN_OCCURENCE]

    @staticmethod
    def load():
        if os.path.isfile(PATH):
            return load(PATH)
        else:
            print('No data available. Initiating data.')
            exp_data = ExperimentalData()
            dump(exp_data, PATH)
