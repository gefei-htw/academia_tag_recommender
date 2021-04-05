from pathlib import Path
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.matutils import unitvec
from gensim.models.doc2vec import TaggedDocument
from academia_tag_recommender.stopwords import stopwordlist
from academia_tag_recommender.definitions import MODELS_PATH

DATA_FOLDER = Path(MODELS_PATH) / 'classifier' / 'multi-label'
WORD2VEC_MODEL_PATH = DATA_FOLDER / 'word2vec'
DOC2VEC_MODEL_PATH = DATA_FOLDER / 'doc2vec'
FASTTEXT_MODEL_PATH = DATA_FOLDER / 'fasttext'

# code parts taken from: https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/ipynb_with_output/Document%20classification%20with%20word%20embeddings%20tutorial%20-%20with%20output.ipynb


def _sent2tokens(sentence):
    tokens = []
    for word in simple_preprocess(sentence):
        if word in stopwordlist:
            continue
        tokens.append(word)
    return tokens


def _word2tokens(document, flat=True):
    sentences = []
    for sentence in sent_tokenize(document, language='english'):
        sentence = _sent2tokens(sentence)
        if flat:
            sentences = sentences + sentence
        else:
            sentences.append(sentence)
    return sentences


class Word2Tok:
    def __init__(self, data, flat=True):
        self.data = data
        self.flat = flat

    def __iter__(self):
        for document in self.data:
            sentences = _word2tokens(document[0], self.flat)
            if self.flat:
                yield sentences
            else:
                for sentence in sentences:
                    yield sentence


class Doc2Tagged:
    def __init__(self, data, tag=False):
        self.data = data
        self.tag = tag

    def __iter__(self):
        for i, document in enumerate(self.data):
            tokens = _word2tokens(document[0])
            if self.tag:
                yield TaggedDocument(tokens, [i])
            else:
                yield tokens


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.word_vec(word, use_norm=True))
            all_words.add(wv.vocab[word].index)

    mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, samples):
    return np.vstack([word_averaging(wv, sample) for sample in samples])


def doc2vector(model, samples):
    return [model.infer_vector(sample) for sample in samples]
