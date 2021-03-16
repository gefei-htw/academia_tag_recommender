"""This module handles preprocessing definitions."""
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec, FastText
from academia_tag_recommender.preprocessor import BasicPreprocessor

from academia_tag_recommender.tokenizer import BasicTokenizer, EnglishStemmer, LancasterStemmer, Lemmatizer, PorterStemmer


class PreprocessingDefinition:
    """This stores a definition of the preprocessing steps.
    """

    def __init__(self, vectorizer, tokenizer, preprocessor, stopwords, n_gram, dim_reduce):
        self.vectorizer = VectorizerDefinition(vectorizer)
        self.tokenizer = TokenizerDefinition(tokenizer)
        self.preprocessor = PreprocessorDefinition(preprocessor)
        self.stopwords = stopwords
        self.n_gram = n_gram
        self.dim_reduce = dim_reduce

    @classmethod
    def from_uri(cls, uri):
        matches = re.findall(r'=([\w,\d]*)', uri)
        vectorizer, tokenizer, preprocessor, stopwords, n_gram, dim_reduce = matches
        return cls(vectorizer, tokenizer, preprocessor, stopwords, n_gram, dim_reduce)

    def __str__(self):
        """
        docstring
        """
        return 'v={}&t={}&p={}&s={}&n={}&dim={}'.format(self.vectorizer.short_name, self.tokenizer.short_name, self.preprocessor.short_name, self.stopwords, self.n_gram, self.dim_reduce)


class Definition:

    classes = {}

    def __init__(self, name):
        self.short_name = name
        self.class_ = self.get_class(name)

    def __str__(self):
        return '{}: {}'.format(self.short_name, self.class_)

    def get_class(self, name):
        if name in self.classes.keys():
            return self.classes[name]
        raise KeyError('Given key is not a valid class')


class VectorizerDefinition(Definition):

    classes = {
        'count': CountVectorizer,
        'tfidf': TfidfVectorizer,
        'word2vec': Word2Vec,
        'doc2vec': Doc2Vec,
        'fasttext': FastText}

    def __init__(self, name):
        super().__init__(name)


class TokenizerDefinition(Definition):

    classes = {
        'None': None,
        'basic': BasicTokenizer,
        'english': EnglishStemmer,
        'porter': PorterStemmer,
        'lancaster': LancasterStemmer,
        'lemmatizer': Lemmatizer}

    def __init__(self, name):
        super().__init__(name)


class PreprocessorDefinition(Definition):

    classes = {
        'None': None,
        'basic': BasicPreprocessor}

    def __init__(self, name):
        super().__init__(name)
