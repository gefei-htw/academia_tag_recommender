"""Handles documents."""

import nltk
import matplotlib.pyplot as plt
import numpy as np

from academia_tag_recommender.data import questions


def _format_tags(tags):
    """Reformats a string of tags.

    Extracts a list of strings from a string in diamond notation.

    :param tags: Tags in diamond notation as one string
    :type tags: str
    :return: A list of tags
    :rtype: list(str)
    """
    tags_ = tags.replace('<', '').split('>')
    return tags_[0:(len(tags_)-1)]


class Document:
    """This is a class representation of a document.

    :param title: The documents title
    :type title: str
    :param body: The documents body
    :type body: str
    :param text: A combination of the documents title and body
    :type text: str
    :param tags: A list of the documents tags
    :type tags: list(str)
    """

    def __init__(self, title, body, tags):
        """Constructor method"""
        self.title = title
        self.body = body
        self.text = title + body
        self.tags = _format_tags(tags)

    def __repr__(self):
        """Representation method"""
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)

    def __str__(self):
        """String method"""
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)


def documents():
    """Return documents from the stack exchange data dump.

    :return: A list of documents
    :rtype: list(Document)
    """
    return [Document(document.attrib['Title'], document.attrib['Body'],
                     document.attrib['Tags']) for document in questions()]
