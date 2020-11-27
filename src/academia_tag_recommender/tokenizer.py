from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as NLTKLemmatizer
from nltk.stem.snowball import EnglishStemmer as NLTKEnglishStemmer, PorterStemmer as NLTKPorterStemmer
from nltk.stem.lancaster import LancasterStemmer as NTLKLancasterStemmer


class BasicTokenizer:
    """Splits text into tokens that are minimum 2 characters long.

        :param text: The unprocessed string
        :type text: str
        :return: The list of tokens
        :rtype: list(str)
        """

    def __call__(self, text):
        tokens = word_tokenize(text)
        return [word for word in tokens if len(word) > 1]


class EnglishStemmer:
    """Splits text into tokens using the :class:`BasicTokenizer` and stems them using the :class:`nltk.stem.snowball.EnglishStemmer`.

        :param text: The unprocessed string
        :type text: str
        :return: The list of tokens
        :rtype: list(str)
        """

    def __self__(self):
        self.tokenizer = BasicTokenizer()

    def __call__(self, text):
        tokens = self.tokenizer(text)
        tokens = [NLTKEnglishStemmer().stem(word) for word in tokens]
        return tokens


class PorterStemmer:
    """Splits text into tokens using the :class:`BasicTokenizer` and stems them using the :class:`nltk.stem.snowball.PorterStemmer`.

        :param text: The unprocessed string
        :type text: str
        :return: The list of tokens
        :rtype: list(str)
        """

    def __self__(self):
        self.tokenizer = BasicTokenizer()

    def __call__(self, text):
        tokens = self.tokenizer(text)
        tokens = [NLTKPorterStemmer().stem(word) for word in tokens]
        return tokens


class LancasterStemmer:
    """Splits text into tokens using the :class:`BasicTokenizer` and stems them using the :class:`nltk.stem.lancaster.LancasterStemmer`.

        :param text: The unprocessed string
        :type text: str
        :return: The list of tokens
        :rtype: list(str)
        """

    def __self__(self):
        self.tokenizer = BasicTokenizer()

    def __call__(self, text):
        tokens = self.tokenizer(text)
        tokens = [NTLKLancasterStemmer().stem(word) for word in tokens]
        return tokens


class Lemmatizer:
    """Splits text into tokens using the :class:`BasicTokenizer` and lemmatizes them using the :class:`nltk.stem.WordNetLemmatizer`.

        :param text: The unprocessed string
        :type text: str
        :return: The list of tokens
        :rtype: list(str)
        """

    def __self__(self):
        self.tokenizer = BasicTokenizer()

    def __call__(self, text):
        tokens = self.tokenizer(text)
        tokens = [NLTKLemmatizer().lemmatize(word) for word in tokens]
        return tokens
