"""This module handles classifier calculation."""
from academia_tag_recommender.definitions import MODELS_PATH
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from pathlib import Path
import random
import numpy as np

DATA_FOLDER = Path(MODELS_PATH) / 'experimental_classifier' / 'classwise'

SAMPLE_RATIO = 1 / 25
RANDOM_STATE = 0
random.seed(RANDOM_STATE)
scorer = make_scorer(recall_score)
k_fold = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)


class ClasswiseClassifier:
    """The BR Classwise Classifier that is capable of of grid search and undersampling.

    Attributes:
        name: The experimental classifiers name as :class:`str`.
        path: The experimental classifiers path on the disc as :class:`pathlib.Path`.
        classifier_options: The options for grid search as :class:`list(ClassifierOption)`.
        path: The path where the individual base classifiers are stored as :class:`pathlib.Path`.
        undersample: If True undersampling is used.
    """

    def __init__(self, name, classifier_options, folder_path, undersample=False):
        self.name = name
        self.classifier_options = classifier_options
        self.path = DATA_FOLDER / folder_path
        Path.mkdir(self.path, exist_ok=True)
        self.undersample = undersample

    def fit(self, X, y):
        """Fit classifier to given data.

        Args:
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The classifier as :class:`ClasswiseClassifier`.
        """
        self._clfs = []
        for y_i, _ in enumerate(y[0]):
            y_train = y[:, y_i]
            if self.undersample:
                X_sample, y_sample = self._undersample(X, y_train)
            else:
                X_sample, y_sample = X, y_train
            clf = self._choose_classifier(X_sample, y_sample)
            path = self._dump_clf(clf, y_i)
            self._clfs.append(path)
        return self

    def _positive_samples(self, X, y):
        """Extract only positive samples.

        Args:
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The positive samples as :class:`list`.
        """
        i_positive = [i for i, _ in enumerate(X) if y[i]]
        return random.sample(i_positive, len(i_positive))

    def _negative_samples(self, X, y, n_pos):
        """Extract negative samples with adjusted ratio to positive samples.

        Args:
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The negative samples as :class:`list`.
        """
        i_negative = [i for i, _ in enumerate(X) if not y[i]]
        n_neg = min(len(i_negative), round(n_pos / SAMPLE_RATIO))
        return random.sample(i_negative, n_neg)

    def _undersample(self, X, y):
        """Reduce X and y to an adjusted ratio of positive and negative samples.

        Args:
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The adjusted samples as :class:`list`.
        """
        i_pos = self._positive_samples(X, y)
        i_neg = self._negative_samples(X, y, len(i_pos))
        i = i_pos + i_neg
        return np.array(X)[i], np.array(y)[i]

    def _choose_classifier(self, X, y):
        """Find the best fitting classifier.

        Args:
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The best classifier.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.25, random_state=RANDOM_STATE)
        clfs = [self._fit_clf(clf_option, X_train, y_train)
                for clf_option in self.classifier_options]
        clf = self._get_best_clf(clfs, X_test, y_test)
        return clf

    def _fit_clf(self, clf_option, X, y):
        """Train classifiers as defined by the option.

        Args:
            clf_option:
                The classifier that should be trained as :class:`ClassifierOption`.
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The trained classifier.
        """
        if clf_option.grid_search:
            return GridSearchCV(clf_option.clf, clf_option.parameter, cv=k_fold, scoring=scorer).fit(X, y).best_estimator_
        else:
            return clf_option.clf.fit(X, y)

    def _get_best_clf(self, clfs, X, y):
        """Calculate scores for each classifier and return best.

        Args:
            clfs:
                The classifiers to choose from as :class:`list`.
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The best classifier.
        """
        clf_scores = [(clf, self._score_clf(clf, X, y)) for clf in clfs]
        clf = sorted(clf_scores, key=lambda x: x[1], reverse=True)[0][0]
        return clf

    def _score_clf(self, clf, X, y):
        """Calculate score using the predicted labels by given classifier.

        Args:
            clfs:
                The classifiers to use.
            X: 
                The samples as :class:`list`.
            y: 
                The label data as :class:`list`.

        Returns:
            The score as :class:`float`.
        """
        prediction = clf.predict(X)
        score = recall_score(y, prediction)
        return score

    def _dump_clf(self, clf, i):
        """Store a classifier on the disc.

         Args:
            clfs:
                The classifiers to store.
            i: 
                Number of the label the classifier handles as :class:`int`.

        Returns:
            The path where the classifier was stored as :class:`joblib.Path`.
        """
        path = self.path / (self.name + '_classifier_' + str(i) + '.joblib')
        dump(clf, path)
        return path

    def predict(self, X):
        """Predict labels based on X.

        Args:
            X: 
                The samples as :class:`list`.

        Returns:
            The prediction as :class:`list`.
        """
        prediction = []
        for path in self._clfs:
            clf = load(path)
            prediction.append(clf.predict(X))
        return np.transpose(prediction)

    def __str__(self):
        return self.name


class ClassifierOption:
    """A classifier and optional gridsearch parameters.

    Attributes:
        clf: The classifier.
        grid_search: If True gridsearch will be used.
        parameter: The parameter to test while gridsearching.
    """

    def __init__(self, clf, grid_search=False, parameter={}):
        self.clf = clf
        self.grid_search = grid_search
        self.parameter = parameter
