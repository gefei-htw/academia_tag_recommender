from pathlib import Path
import xml.etree.ElementTree as ET
from academia_tag_recommender.definitions import DATA_PATH

data_folder = Path(DATA_PATH + '/raw')

tag_tree = ET.parse(data_folder / 'Tags.xml')
tags = tag_tree.getroot()

users_tree = ET.parse(data_folder / 'Users.xml')
users = users_tree.getroot()

post_tree = ET.parse(data_folder / 'Posts.xml')
posts = post_tree.getroot()

questions = [post for post in posts if post.attrib['PostTypeId'] == '1']

comment_tree = ET.parse(data_folder / 'Comments.xml')
comments = comment_tree.getroot()

vote_tree = ET.parse(data_folder / 'Votes.xml')
votes = vote_tree.getroot()
