"""Handles data imports."""
from pathlib import Path
import xml.etree.ElementTree as ET
from academia_tag_recommender.documents import Document
from academia_tag_recommender.definitions import DATA_PATH

_DATA_FOLDER = Path(DATA_PATH + '/raw')

_tag_tree = ET.parse(_DATA_FOLDER / 'Tags.xml')
_users_tree = ET.parse(_DATA_FOLDER / 'Users.xml')
_post_tree = ET.parse(_DATA_FOLDER / 'Posts.xml')
_comment_tree = ET.parse(_DATA_FOLDER / 'Comments.xml')
_vote_tree = ET.parse(_DATA_FOLDER / 'Votes.xml')


tags = _tag_tree.getroot()

users = _users_tree.getroot()

posts = _post_tree.getroot()

questions = [post for post in posts() if post.attrib['PostTypeId'] == '1']

documents = [Document(document.attrib['Title'], document.attrib['Body'],
                      document.attrib['Tags']) for document in questions]

comments = _comment_tree.getroot()

votes = _vote_tree.getroot()
