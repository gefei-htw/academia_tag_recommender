"""This module imports the stopwordlist.

Attributes:
    stopwordlist: The list of stopwords as :class:`list` of :class:`str`
"""

from pathlib import Path
from academia_tag_recommender.definitions import DATA_PATH

_DATA_FOLDER = Path(DATA_PATH + '/external')


def _stopwordlist():
    """Return a stopwordlist.

    Returns:
        A list of stopwords as :class:`list` of :class:`str`.
    """
    with open(_DATA_FOLDER / "stopwordlist") as swlist:
        sws = [line.partition('|')[0].rstrip() for line in swlist]
        sws = [word for word in sws if word]
        return sws


stopwordlist = _stopwordlist()
