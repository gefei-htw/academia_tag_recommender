"""This module handles classifier calculation."""
from gensim.models import Word2Vec, FastText, Doc2Vec
from numpy import vectorize
from academia_tag_recommender.definitions import MODELS_PATH
from joblib import load
from pathlib import Path
import re

from academia_tag_recommender.embedded_data import Doc2Tagged, Word2Tok, word_averaging_list
from academia_tag_recommender.experiments.experimental_classifier import ExperimentalClassifier
from academia_tag_recommender.preprocessing_definition import PreprocessingDefinition, VectorizerDefinition
from academia_tag_recommender.test_train_data import get_all_labels, get_vectorizer, get_dimension_reducer

DIM_PATH = Path(MODELS_PATH) / 'dimension_reduction'
EMB_PATH = Path(MODELS_PATH) / 'classifier' / 'multi-label'
PREPROCESSOR_PATH_COUNT = DIM_PATH / \
    'v=count&t=basic&p=basic&s=english&n=1,1&dim=TruncatedSVD.joblib'
PREPROCESSOR_PATH_TFIDF = DIM_PATH / \
    'v=tfidf&t=basic&p=basic&s=english&n=1,1&dim=TruncatedSVD.joblib'
PREPROCESSOR_PATH_WORD2VEC = EMB_PATH / 'word2vec' / '00vectorizer100.model'
PREPROCESSOR_PATH_DOC2VEC = EMB_PATH / 'doc2vec' / 'vectorizer100.model'
PREPROCESSOR_PATH_FASTTEXT = EMB_PATH / 'fasttext' / '0vectorizer100.model'

PREPROCESSOR_PATHS = [PREPROCESSOR_PATH_COUNT, PREPROCESSOR_PATH_TFIDF,
                      PREPROCESSOR_PATH_WORD2VEC, PREPROCESSOR_PATH_DOC2VEC, PREPROCESSOR_PATH_FASTTEXT]

CLASSIFIER_PATH_PREFIX = Path(MODELS_PATH) / \
    'classifier' / 'multi-label' / 'classwise'


class Recommender:
    """This recommender recommends tags for a given question.
    """

    def __init__(self, classifier_path):
        self.transformer = self._get_transformer(classifier_path)
        preprocessor = self._load_preprocessor(
            self._get_preprocessor_path())
        classifier = load(CLASSIFIER_PATH_PREFIX / classifier_path)
        self.classifier = ExperimentalClassifier(preprocessor, classifier)
        self.labels = get_all_labels()

    def _get_transformer(self, classifier_path):
        return re.search(r'v=(\w*)', classifier_path).group(1)

    def _get_preprocessor_path(self):
        return next(path for path in PREPROCESSOR_PATHS if self.transformer in path.as_uri())

    def _load_preprocessor(self, path):
        if self.transformer in ['tfidf', 'count']:
            preprop_def = PreprocessingDefinition.from_uri(str(path))
            vectorizer = get_vectorizer(preprop_def)
            dim_reducer = get_dimension_reducer(preprop_def)
            return BagOfWordsPreprocessor(vectorizer, dim_reducer)
        elif self.transformer == 'word2vec':
            return Word2Vec.load(str(path))
        elif self.transformer == 'fasttext':
            model = FastText.load(str(path))
            wv = model.wv
            wv.init_sims(replace=True)
            return wv
        elif self.transformer == 'doc2vec':
            return Doc2Vec.load(str(path))

    def recommend_tags(self, title, body):
        text = [title + ' ' + body]
        preprocessed = self.classifier.transform(text)
        predicted_tags = self.classifier.predict(preprocessed)
        return [self.labels[i] for i, label in enumerate(predicted_tags) if label == 1]


class BagOfWordsPreprocessor:

    def __init__(self, vectorizer, dim_reducer):
        self.vectorizer = vectorizer
        self.dim_reducer = dim_reducer

    def transform(self, data):
        transformed = self.vectorizer.transform(data)
        reduced = self.dim_reducer.transform(transformed)
        return reduced
