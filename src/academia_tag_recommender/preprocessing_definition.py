"""This module handles preprocessing definitions."""


class PreprocessingDefinition:
    """This stores a definition of the preprocessing steps.
    """

    def __init__(self, vectorizer, tokenizer, preprocessor, stopwords, n_gram, dim_reduce):
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.stopwords = stopwords
        self.n_gram = n_gram
        self.dim_reduce = dim_reduce

    def __str__(self):
        """
        docstring
        """
        return 'v={}&t={}&p={}&s={}&n={}&dim={}'.format(self.vectorizer, self.tokenizer, self.preprocessor, self.stopwords, self.n_gram, self.dim_reduce)
