from pathlib import Path
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.matutils import unitvec
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from academia_tag_recommender.stopwords import stopwordlist
from academia_tag_recommender.definitions import MODELS_PATH

DATA_FOLDER = Path(MODELS_PATH) / 'classifier' / 'multi-label'
WORD2VEC_MODEL_PATH = DATA_FOLDER / 'word2vec'
DOC2VEC_MODEL_PATH = DATA_FOLDER / 'doc2vec'
FASTTEXT_MODEL_PATH = DATA_FOLDER / 'fasttext'


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
    def __init__(self, data, y=None):
        self.data = data
        self.y = y

    def __iter__(self):
        for i, document in enumerate(self.data):
            tokens = _word2tokens(document[0])
            if isinstance(self.y, np.ndarray):
                yield TaggedDocument(tokens, self.y[i])
            else:
                yield tokens


# code taken from: https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/ipynb_with_output/Document%20classification%20with%20word%20embeddings%20tutorial%20-%20with%20output.ipynb
def _word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.word_vec(word, use_norm=True))
            all_words.add(wv.vocab[word].index)

    mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def _word_averaging_list(wv, samples):
    return np.vstack([_word_averaging(wv, sample) for sample in samples])


def word2vec(X_train, X_test, vector_size=100, sg=False):
    X_train_word2sen = Word2Tok(X_train, flat=False)
    X_train_word2tok = Word2Tok(X_train)
    X_test_word2tok = Word2Tok(X_test)

    wv = _get_word2vec_wv(X_train_word2sen, vector_size, sg)
    return _word_averaging_list(wv, X_train_word2tok), _word_averaging_list(wv, X_test_word2tok)


def _get_word2vec_wv(sentences, vector_size, skip_gram):
    sg = 1 if skip_gram else 0
    path = WORD2VEC_MODEL_PATH / \
        (str(sg) + 'vectorizer' + str(vector_size) + '.model')
    if Path.is_file(path):
        model = Word2Vec.load(str(path))
    else:
        model = Word2Vec(sentences=sentences, size=vector_size, sg=sg)
        model.save(str(path))
    wv = model.wv
    wv.init_sims(replace=True)
    return wv


def doc2vec(X_train, X_test, y, vector_size=100):
    X_train_doc2tok = Doc2Tagged(X_train, y)
    X_test_doc2tok = Doc2Tagged(X_test)

    model = _get_doc2vec_model(X_train_doc2tok, vector_size)
    return _doc2vector(model, [sample.words for sample in X_train_doc2tok]), _doc2vector(model, X_test_doc2tok)


def _get_doc2vec_model(tokens, vector_size):
    path = DOC2VEC_MODEL_PATH / ('vectorizer' + str(vector_size) + '.model')
    if Path.is_file(path):
        model = Doc2Vec.load(str(path))
    else:
        model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=20)
        model.build_vocab(tokens)
        model.train(tokens, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save(str(path))
    return model


def _doc2vector(model, samples):
    return [model.infer_vector(sample) for sample in samples]


def fasttext2vec(X_train, X_test, vector_size=100):
    X_train_word2sen = Word2Tok(X_train, flat=False)
    X_train_word2tok = Word2Tok(X_train)
    X_test_word2tok = Word2Tok(X_test)

    wv = _get_fasttext_wv(X_train_word2sen, vector_size)
    return _word_averaging_list(wv, X_train_word2tok), _word_averaging_list(wv, X_test_word2tok)


def _get_fasttext_wv(sentences, vector_size):
    path = FASTTEXT_MODEL_PATH / ('vectorizer' + str(vector_size) + '.model')
    if Path.is_file(path):
        model = FastText.load(str(path))
    else:
        model = FastText(size=vector_size, window=3, min_count=2)
        model.build_vocab(sentences=sentences)
        model.train(sentences=sentences,
                    total_examples=model.corpus_count, epochs=20)
        model.save(str(path))
    wv = model.wv
    wv.init_sims(replace=True)
    return wv
