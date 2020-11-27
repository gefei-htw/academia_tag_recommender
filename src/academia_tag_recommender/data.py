"""Handles data imports."""

from pathlib import Path
import xml.etree.ElementTree as ET
from academia_tag_recommender.definitions import DATA_PATH

_DATA_FOLDER = Path(DATA_PATH + '/raw')

_tag_tree = ET.parse(_DATA_FOLDER / 'Tags.xml')
_users_tree = ET.parse(_DATA_FOLDER / 'Users.xml')
_post_tree = ET.parse(_DATA_FOLDER / 'Posts.xml')
_comment_tree = ET.parse(_DATA_FOLDER / 'Comments.xml')
_vote_tree = ET.parse(_DATA_FOLDER / 'Votes.xml')


def tags():
    """Return tags from the stack exchange data dump.

    :return: An ElementTree of the tags xml
    :rtype: ElementTree
    """
    return _tag_tree.getroot()


def users():
    """Return users from the stack exchange data dump.

    :return: An ElementTree of the users xml
    :rtype: ElementTree
    """
    return _users_tree.getroot()


def posts():
    """Return posts from the stack exchange data dump.

    :return: An ElementTree of the posts xml
    :rtype: ElementTree
    """
    return _post_tree.getroot()


def questions():
    """Return questions from the stack exchange data dump.

    Questions are Posts with PostTypeId == 1

    :return: An ElementTree of the questions xml
    :rtype: ElementTree
    """
    return [post for post in posts() if post.attrib['PostTypeId'] == '1']


def comments():
    """Return comments from the stack exchange data dump.

    :return: An ElementTree of the comments xml
    :rtype: ElementTree
    """
    return _comment_tree.getroot()


def votes():
    """Return votes from the stack exchange data dump.

    :return: An ElementTree of the votes xml
    :rtype: ElementTree
    """
    return _vote_tree.getroot()
