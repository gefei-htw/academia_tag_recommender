import os
from pathlib import Path
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.phrases import Phrases
from academia_tag_recommender.stopwords import stopwordlist
from academia_tag_recommender.embeddings import Doc2Tagged, Word2Tok, doc2vector, word_averaging_list
from academia_tag_recommender.preprocessor import BasicPreprocessor
from academia_tag_recommender.tokenizer import BasicTokenizer, EnglishStemmer, Lemmatizer
from sklearn.decomposition import TruncatedSVD
from academia_tag_recommender.definitions import MODELS_PATH

PATH = Path(MODELS_PATH) / 'transformer'

RANDOM_STATE = 0
MAX_FEATURES = 5993

TRANSFORMER_OPTIONS = {
    'tfidf': TfidfVectorizer,
    'count': CountVectorizer,
    'word2vec': Word2Vec,
    'doc2vec': Doc2Vec,
    'fasttext': FastText
}

TOKENIZER_OPTIONS = {
    'basic': BasicTokenizer,
    'stemmer': EnglishStemmer,
    'lemmatizer': Lemmatizer
}

DIM_REDUCE_OPTIONS = {
    'lsa': TruncatedSVD
}


class Transformer:
    """The transformer to represent texts as a vectors.

    Attributes:
        vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
        vectorizer: The used vectorizer.
    """

    def __init__(self, vectorizer):
        self.vectorizer_short_name = vectorizer
        self.vectorizer = TRANSFORMER_OPTIONS[vectorizer]

    def __str__(self):
        return 'v={}'.format(self.vectorizer_short_name)

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    @classmethod
    def load(cls, vectorizer_short_name):
        """Loads an existing Transformer instance from disc or creates new if none exists.

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.

        Returns:
            The configured class:`Transformer` instance.
        """
        path = cls._to_path(vectorizer_short_name)
        if os.path.isfile(path):
            return load(path)
        else:
            print('Transformer not available. Initiating data.')
            return cls(vectorizer_short_name)

    @staticmethod
    def _to_path(vectorizer_short_name):
        """Converts vectorizer name into path name

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.

        Returns:
            The Path of the vector as :class:`pathlib.Path`.
        """
        return PATH / 'v={}.joblib'.format(vectorizer_short_name)


class BagOfWordsTransformer(Transformer):
    """The transformer to represent texts as bag of word vectors.

    Attributes:
        vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
        vectorizer: The used vectorizer.
        tokenizer_short_name: The short name of the used tokenizer as :class:`str`.
        tokenizer: The used tokenizer.
        dimension_reduction_short_name: The short name of the used dimension reduction as :class:`str`.
        dimension_reduction: The used dimension reduction.
        path: The path where the transformer is stored on the disc as :class:`pathlib.Path`.
    """

    def __init__(self, vectorizer, tokenizer, dimension_reduction):
        """

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
            tokenizer_short_name: The short name of the used tokenizer as :class:`str`.
            dimension_reduction_short_name: The short name of the used dimension reduction as :class:`str`.
        """
        self.tokenizer_short_name = tokenizer
        self.tokenizer = TOKENIZER_OPTIONS[tokenizer]
        self.vectorizer_short_name = vectorizer
        self.vectorizer = TRANSFORMER_OPTIONS[vectorizer](
            min_df=2, tokenizer=self.tokenizer(), preprocessor=BasicPreprocessor(), stop_words=stopwordlist, ngram_range=(1, 1))
        self.dimension_reduction_short_name = dimension_reduction
        self.dimension_reduction = DIM_REDUCE_OPTIONS[dimension_reduction](
            n_components=MAX_FEATURES, random_state=RANDOM_STATE)
        self.path = BagOfWordsTransformer._to_path(
            self.vectorizer_short_name, self.tokenizer_short_name, self.dimension_reduction_short_name)

    def __str__(self):
        return 'v={}&t={}&dim_reduce={}'.format(self.vectorizer_short_name, self.tokenizer_short_name, self.dimension_reduction_short_name)

    def fit(self, X):
        """Creates a transformer based on X.

        Args:
            X: The samples data as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        self.path = BagOfWordsTransformer._to_path(
            self.vectorizer_short_name, self.tokenizer_short_name, self.dimension_reduction_short_name)
        if not os.path.isfile(self.path):
            features = self.vectorizer.fit_transform(X)
            reduced = self.dimension_reduction.fit_transform(features)
            dump(self, self.path)
            return reduced
        else:
            return self.transform(X)

    def transform(self, X):
        """Transforms X.

        Args:
            X: The samples data as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        features = self.vectorizer.transform(X)
        return self.dimension_reduction.transform(features)

    @classmethod
    def load(cls, vectorizer_short_name, tokenizer_short_name, dimension_reduction_short_name):
        """Loads an existing BagOfWordsTransformer instance from disc or creates new if none exists.

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
            tokenizer_short_name: The short name of the used tokenizer as :class:`str`.
            dimension_reduction_short_name: The short name of the used dimension reduction as :class:`str`.

        Returns:
            The configured :class:`BagOfWordsTransformer` instance.
        """
        path = cls._to_path(
            vectorizer_short_name, tokenizer_short_name, dimension_reduction_short_name)
        if os.path.isfile(path):
            return load(path)
        else:
            print('Transformer not available. Initiating data.')
            return cls(
                vectorizer_short_name, tokenizer_short_name, dimension_reduction_short_name)

    @staticmethod
    def _to_path(vectorizer_short_name, tokenizer_short_name, dimension_reduction_short_name):
        """Converts vectorizer name, tokenizer name and dimensionreduction name into path name.

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
            tokenizer_short_name: The short name of the used tokenizer as :class:`str`.
            dimension_reduction_short_name: The short name of the used dimension reduction as :class:`str`.

        Returns:
            The Path as :class:`pathlib.Path`.
        """
        return PATH / 'v={}&t={}&dim_reduce={}.joblib'.format(vectorizer_short_name, tokenizer_short_name, dimension_reduction_short_name)


class EmbeddingTransformer(Transformer):
    """The transformer to represent texts as embedding vectors.

    Attributes:
        vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
        path: The path where the transformer is stored on the disc as :class:`pathlib.Path`.
    """

    def __init__(self, vectorizer, vector_size=100):
        self.vectorizer_short_name = vectorizer
        self.vector_size = vector_size
        self.path = EmbeddingTransformer._to_path(vectorizer, vector_size)

    def __str__(self):
        return 'v={}&size={}'.format(self.vectorizer_short_name, self.vector_size)

    def fit(self, X, bigramify=False):
        """Creates a transformer based on X.

        Args:
            X: The samples data as :class:`list`.
            bigramify: If True, bigrams will be included.

        Returns:
            The transformed samples as :class:`list`.
        """
        if not hasattr(self, 'vectorizer'):
            self._prepare(X, bigramify)
        transformed = self.transform(X)
        dump(self, self.path)
        return transformed

    def _prepare(self, X, bigramify):
        raise NotImplementedError

    def transform(self, X, bigramify=False):
        """Transforms X.

        Args:
            X: The samples data as :class:`list`.
            bigramify: If True, bigrams will be included.

        Returns:
            The transformed samples as :class:`list`.
        """
        X = [[x] for x in X]
        if bigramify:
            X = self.bigram_transformer[X]
        X_word2tok = Word2Tok(X)
        return word_averaging_list(self.vectorizer, X_word2tok)

    @classmethod
    def load(clf, vectorizer_short_name, vector_size=100):
        """Loads an existing Transformer instance from disc or creates new if none exists.

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
            vector_size: The vector size of resulting vectorizer as :class:`int`.

        Returns:
            The configured :class:`EmbeddingTransformer` instance.
        """
        path = clf._to_path(vectorizer_short_name, vector_size)
        if os.path.isfile(path):
            return load(path)
        else:
            print('Transformer not available. Initiating data.')
            return clf(vectorizer_short_name, vector_size)

    @staticmethod
    def _to_path(vectorizer_short_name, vector_size):
        """Converts vectorizer name and vector size into path name.

        Args:
            vectorizer_short_name: The short name of the used vectorizer as :class:`str`.
            vector_size: The vector size of resulting vectorizer as :class:`int`.

        Returns:
            The Path as :class:`pathlib.Path`.
        """
        return PATH / 'v={}&size={}.joblib'.format(vectorizer_short_name, vector_size)


class Word2VecTransformer(EmbeddingTransformer):
    """The transformer to represent texts as word2vec vectors.

    Attributes:
        vectorizer: The word2vec vectorizer as :class:`gensim.models.keyedvectors.Word2VecKeyedVectors`.
        bigram_transformer: The sentence to bigram transformer as :class:`gensim.models.phrases.Phrases`.
    """

    def _prepare(self, X, bigramify):
        """Trains a word2vec transformer based on X using bigrams if specified.

        Args:
            X: The samples data as :class:`list`.
            bigramify: If True, bigrams will be included.
        """
        X = [[x] for x in X]
        sentences = Word2Tok(X, flat=False)
        if bigramify:
            self.bigram_transformer = Phrases(sentences, min_count=1)
            sentences = self.bigram_transformer[sentences]
        model = Word2Vec(sentences=sentences, size=self.vector_size)
        self.vectorizer = model.wv
        del model
        self.vectorizer.init_sims(replace=True)


class FastTextTransformer(EmbeddingTransformer):
    """The transformer to represent texts as fasttext vectors.

    Attributes:
        vectorizer: The word2vec vectorizer as :class:`gensim.models.keyedvectors.FastTextKeyedVectors`.
        bigram_transformer: The sentence to bigram transformer as :class:`gensim.models.phrases.Phrases`.

    """

    def _prepare(self, X, bigramify):
        """Trains a fasttext transformer based on X using bigrams if specified.

        Args:
            X: The samples data as :class:`list`.
            bigramify: If True, bigrams will be included.
        """
        X = [[x] for x in X]
        sentences = Word2Tok(X, flat=False)
        if bigramify:
            self.bigram_transformer = Phrases(sentences, min_count=1)
            sentences = self.bigram_transformer[sentences]
        model = FastText(size=self.vector_size, window=3, min_count=2)
        model.build_vocab(sentences=sentences)
        model.train(sentences=sentences,
                    total_examples=model.corpus_count, epochs=20)
        self.vectorizer = model.wv
        del model
        self.vectorizer.init_sims(replace=True)


class Doc2VecTransformer(EmbeddingTransformer):
    """The transformer to represent texts as doc2vec vectors.

    Attributes:
        vectorizer: The word2vec vectorizer as :class:`gensim.models.doc2vec.Doc2Vec`.
    """

    def _prepare(self, X):
        """Trains a doc2vec transformer based on X.

        Args:
            X: The samples data as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        X = [[x] for x in X]
        tokens = Doc2Tagged(X, tag=True)
        self.vectorizer = Doc2Vec(
            vector_size=self.vector_size, min_count=2, epochs=20)
        self.vectorizer.build_vocab(tokens)
        self.vectorizer.train(tokens, total_examples=self.vectorizer.corpus_count,
                              epochs=self.vectorizer.epochs)
        return doc2vector(self.vectorizer, [sample.words for sample in tokens])

    def transform(self, X):
        """Transforms X.

        Args:
            X: The samples data as :class:`list`.

        Returns:
            The transformed samples as :class:`list`.
        """
        X = [[x] for x in X]
        X_doc2tok = Doc2Tagged(X)
        return doc2vector(self.vectorizer, X_doc2tok)
