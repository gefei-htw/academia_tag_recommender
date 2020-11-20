from pathlib import Path
import xml.etree.ElementTree as ET

data_folder = Path("../data/raw/")

tagTree = ET.parse(data_folder / 'Tags.xml')
tags = tagTree.getroot()

usersTree = ET.parse(data_folder / 'Users.xml')
users = usersTree.getroot()

postTree = ET.parse(data_folder / 'Posts.xml')
posts = postTree.getroot()

questions = list(filter(lambda x: x.attrib['PostTypeId'] == '1', posts))

commentTree = ET.parse(data_folder / 'Comments.xml')
comments = commentTree.getroot()

voteTree = ET.parse(data_folder / 'Votes.xml')
votes = voteTree.getroot()
