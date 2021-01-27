import numpy as np
from joblib import dump, load
from pathlib import Path
from academia_tag_recommender.definitions import MODELS_PATH
from academia_tag_recommender.vectorizer_computation import get_vect_feat_with_params
from academia_tag_recommender.data import documents
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

data_folder = Path(MODELS_PATH + '/dimension_reduction')

texts = [document.text for document in documents]


def get_test_train_data(X, y, split=0.25, multi=True):

    if multi:
        X, y = remove_classes_with_few_occurences(X, y)
    else:
        y = y[:, 7]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=0)
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


def get_idx_with_few_occurences(y, n):
    class_occurences = y.sum(axis=0)
    lable_idx_to_remove = []
    for index, class_occurences in enumerate(class_occurences):
        if class_occurences < n:
            lable_idx_to_remove.append(index)

    data_idx_to_remove = []
    for index, lable in enumerate(y):
        for lable_idx in lable_idx_to_remove:
            if lable[lable_idx] == 1:
                data_idx_to_remove.append(index)

    return [lable_idx_to_remove, data_idx_to_remove]


def remove_classes_with_few_occurences(X, y, n=3):
    [lable_idx_to_remove,
        data_idx_to_remove] = get_idx_with_few_occurences(y, n)

    while len(lable_idx_to_remove) > 0 or len(data_idx_to_remove) > 0:
        y = np.delete(y, lable_idx_to_remove, axis=1)
        y = np.delete(y, data_idx_to_remove, axis=0)
        X = np.delete(X, data_idx_to_remove, axis=0)
        [lable_idx_to_remove,
            data_idx_to_remove] = get_idx_with_few_occurences(y, n)

    return (X, y)
