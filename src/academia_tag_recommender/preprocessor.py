import re

html_re = re.compile(r'<[^>]+>')
punctuation_re = re.compile(r'[^\w\s]')

def BasicPreprocessor(text):
    return punctuation_re.sub(' ', html_re.sub(' ', text.lower()))