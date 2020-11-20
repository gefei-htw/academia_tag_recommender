from pathlib import Path
import xml.etree.ElementTree as ET

data_folder = Path("../data/external/")


def read_stopwordlist():
    swlist = open(data_folder / "stopwordlist")
    sws = [line.partition('|')[0].rstrip() for line in swlist]
    sws = [word for word in sws if word]
    swlist.close()
    return sws


stopwordlist = read_stopwordlist()
