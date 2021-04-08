"""This module holds functionality for embedding transformations
Original code is from:
RARE Technologies (https://github.com/RaRe-Technologies)
Authors:    tmylk, mik8142
Link:       https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/ipynb_with_output/Document%20classification%20with%20word%20embeddings%20tutorial%20-%20with%20output.ipynb
Projekt:    movie-plot-by-genre (https://github.com/RaRe-Technologies/movie-plots-by-genre)
File_name:  Document classification with word embeddings tutorial - with output.ipynb
Commit:     152a666 on 17 Jan 2019 
"""
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


def _sent2tokens(sentence):
    """Tokenize sentences by preprocessing and removing stopwords.

    Args:
        sentence: The sentence to tokenize as :class:`str`.

    Returns:
        List of tokens as :class:`list` of :class:`str`.
    """
    tokens = []
    for word in simple_preprocess(sentence):
        if word in stopwordlist:
            continue
        tokens.append(word)
    return tokens


def _word2tokens(document, flat=True):
    """Tokenize document.

    Args:
        document: The document as :class:`str`.
        flat: If True a flat list of tokens will be returned, otherwise sentences will be kept seperate.

    Returns:
        List of tokens or list of lists of tokens
    """
    sentences = []
    for sentence in sent_tokenize(document, language='english'):
        sentence = _sent2tokens(sentence)
        if flat:
            sentences = sentences + sentence
        else:
            sentences.append(sentence)
    return sentences


class Word2Tok:
    """This is a interator that yields sentences from a set of documents.

    Args:
        data: The set of documents as :class:`list` of :class:`str`.
        flat: If True a flat list of tokens will be returned, otherwise sentences will be kept seperate.

    Yields:
        Sentences
    """

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
    """This is a interator that yields either tokens from a set of documents.

    Args:
        data: The set of documents as :class:`list` of :class:`str`.
        tag: If True the tokens will be tagged otherwise not.

    Yields:
        List of tokens or :class:`gensim.models.doc2vec.TaggedDocument`.
    """

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
    """Calculate average word vectors.

    Args:
        wv: The keyed vectors instance to use to get word vectors as :class:`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors`.
        words: The words to transform into vectors as :class:`list` of :class:`str`.

    Returns:
        The averaged vector as :class:`list` of :class:`float`.
    """
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
    """Get a list of averaged word vectors for samples

    Args:
        wv: The keyed vectors instance to use to get word vectors as :class:`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors`.
        samples: The samples as :class:`list`.

    Returns:
        List of averaged vectors as :class:`list` of :class:`list`.
    """
    return np.vstack([word_averaging(wv, sample) for sample in samples])


def doc2vector(model, samples):
    """Infer vectors for samples

    Args:
        model: The instance to use to infer vectors vectors as :class:`gensim.models.Doc2Vec`.
        samples: The samples as :class:`list`.

    Returns:
        The :class:`list` of inferred vectors.
    """
    return [model.infer_vector(sample) for sample in samples]
