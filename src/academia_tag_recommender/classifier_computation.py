from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np
import sys
import time
import os.path
import matplotlib.pyplot as plt
from pathlib import Path
from academia_tag_recommender.definitions import MODELS_PATH
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

data_folder = Path(MODELS_PATH + '/classifier')
results_path = data_folder / 'results.joblib'
results_multi_path = data_folder / 'results_multi.joblib'
clf_path = data_folder / 'clfs'


def binary_confusion_matrix(y, prediction):
    threshold = 0.5
    true_positives = len([1 for i, e in enumerate(
        y) if e == 1 and prediction[i] >= threshold])
    true_negatives = len([1 for i, e in enumerate(
        y) if e == 0 and prediction[i] < threshold])
    false_positives = len([1 for i, e in enumerate(
        y) if e == 0 and prediction[i] >= threshold])
    false_negatives = len([1 for i, e in enumerate(
        y) if e == 1 and prediction[i] < threshold])
    print('{:<15}{:<15}{:<15}'.format('', 'Pred true', 'Pred false'))
    print('{:<15}{:<15}{:<15}'.format(
        'Orig true', true_positives, false_negatives))
    print('{:<15}{:<15}{:<15}'.format(
        'Orig false', false_positives, true_negatives))


def multi_confusion_matrix(y, prediction):
    print(multilabel_confusion_matrix(y, prediction))


def test_prediction(clf, X, y, multi):
    prediction = clf.predict(X)
    plt.show()
    if multi:
        multi_confusion_matrix(y, prediction)
    else:
        binary_confusion_matrix(y, prediction)


def calculate_classifier_results(name, clf, X_train, y_train, X_test, y_test, use_score, multi):
    start = time.time()
    clf_fit = clf.fit(X_train, y_train)
    if use_score:
        score_orig = clf_fit.score(X_train, y_train)
        score_pred = clf_fit.score(X_test, y_test)
    else:
        pred_train = clf_fit.predict(X_train)
        pred_test = clf_fit.predict(X_test)
        score_orig = 1 - mean_squared_error(y_train, pred_train)
        score_pred = 1 - mean_squared_error(y_test, pred_test)
    test_prediction(clf_fit, X_test, y_test, multi)
    end = time.time()
    process_time = end - start
    path = clf_path / name
    dump(clf, path)
    return [name, score_orig, score_pred, process_time]


def add_results(array, results):
    append = True
    for index, result in enumerate(array):
        if results[0] == result[0]:
            array[index] = results
            append = False
    if append:
        array.append(results)
    return array


def test_classifier(name, clf, X_train, y_train, X_test, y_test, use_score=True, multi=False):
    path = None
    if multi:
        path = results_multi_path
    elif not multi:
        path = results_path
    if path:
        array = load(path) if os.path.isfile(path) else []
        array = add_results(array, calculate_classifier_results(
            name, clf, X_train, y_train, X_test, y_test, use_score, multi))
        dump(array, path)


def dump_results(array, multi=False):
    if multi:
        dump(array, results_multi_path)
    else:
        dump(array, results_path)


def load_results(multi=False):
    if multi:
        array = load(results_multi_path)
    else:
        array = load(results_path)
    return array


def load_model(name):
    path = clf_path / name
    clf = load(path)
    return clf
