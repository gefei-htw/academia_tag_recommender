"""This module holds a stopwordlist."""

from pathlib import Path
import xml.etree.ElementTree as ET
from academia_tag_recommender.definitions import DATA_PATH

_DATA_FOLDER = Path(DATA_PATH + '/external')


def stopwordlist():
    """Return a stopwordlist.

    :return: A list of stopwords
    :rtype: list(str)
    """
    with open(_DATA_FOLDER / "stopwordlist") as swlist:
        sws = [line.partition('|')[0].rstrip() for line in swlist]
        sws = [word for word in sws if word]
        return sws
