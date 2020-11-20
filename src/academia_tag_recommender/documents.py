import nltk
import matplotlib.pyplot as plt
import numpy as np

from academia_tag_recommender.data import questions


def formatTags(tags):
    tags_ = tags.replace('<', '').split('>')
    return tags_[0:(len(tags_)-1)]


class Document:

    def __init__(self, title, body, tags):
        self.title = title
        self.body = body
        self.text = title + body
        self.tags = formatTags(tags)

    def __repr__(self):
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)

    def __str__(self):
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)


documents = list(map(lambda x: Document(
    x.attrib['Title'], x.attrib['Body'], x.attrib['Tags']), questions))
