import numpy as np
from joblib import dump, load
from pathlib import Path
from academia_tag_recommender.definitions import MODELS_PATH
from academia_tag_recommender.vectorizer_computation import get_vect_feat_with_params
from academia_tag_recommender.data import documents
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

data_folder = Path(MODELS_PATH + '/dimension_reduction')

texts = [document.text for document in documents]


def get_test_train_data(X, y, split=0.25, multi=True):

    if multi:
        label_indices = get_class_indices_with_enough_occurence(y)
        y = np.array([[column for i, column in enumerate(
            row) if i in label_indices] for row in y])
    else:
        y = y[:, 7]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 0)
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, test_size=split)
    X_train, X_test = scale(X_train, X_test)

    return X_train, X_test, y_train, y_test


def get_X_y(vectorizer, tokenizer, preprocessor, stopwords, n_grams):
    X = get_X(vectorizer, tokenizer, preprocessor, stopwords, n_grams)
    y = get_y()
    return (X, y)


def get_X(vectorizer, tokenizer, preprocessor, stopwords, n_grams):
    [_, features] = get_vect_feat_with_params(
        texts, vectorizer, tokenizer, preprocessor, stopwords, n_grams)
    return features


def get_X_reduced(prepocessing_definition):
    file_name = str(prepocessing_definition) + '.joblib'
    path = data_folder / file_name
    _, X = load(path)
    return X


def fit_labels():
    label = [document.tags for document in documents]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label)
    return (mlb, y)


def get_y():
    _, y = fit_labels()
    return y


def get_all_labels():
    mlb, _ = fit_labels()
    return mlb.classes_


def get_labels(array):
    mlb, _ = fit_labels()
    return [mlb.classes_[i] for i in array]


def get_label(i):
    mlb, _ = fit_labels()
    return mlb.classes_[i]


def scale(train, test):
    scaler = MaxAbsScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return (train, test)


def get_class_indices_with_enough_occurence(y, n=70):
    return [i for i, _ in enumerate(get_all_labels()) if np.sum(y[:, i]) >= n]
