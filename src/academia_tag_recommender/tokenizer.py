from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as NLTKLemmatizer
from nltk.stem.snowball import EnglishStemmer as NLTKEnglishStemmer, PorterStemmer as NLTKPorterStemmer
from nltk.stem.lancaster import LancasterStemmer as NTLKLancasterStemmer
import re

punctuation_re = re.compile(r'[^\w\s]')

def BasicTokenizer(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def EnglishStemmer(text):
    tokens = BasicTokenizer(text)
    tokens = [NLTKEnglishStemmer().stem(word) for word in tokens]         # stem words using the english stemmer
    return tokens

def PorterStemmer(text):
    tokens = BasicTokenizer(text)
    tokens = [NLTKPorterStemmer().stem(word) for word in tokens]          # stem words using the porter stemmer
    return tokens

def LancasterStemmer(text):
    tokens = BasicTokenizer(text)
    tokens = [NTLKLancasterStemmer().stem(word) for word in tokens]       # stem words using the lancaster stemmer
    return tokens

def Lemmatizer(text):
    tokens = BasicTokenizer(text)
    tokens = [NLTKLemmatizer().lemmatize(word) for word in tokens] # lemmatize words
    return tokens    