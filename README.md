# academia_tag_recommender

Tag Recommender for [academia.stackexchange.com](https://academia.stackexchange.com/)

## Table of contents
* [description](#description)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)

## Description

This project aims to recommend tags for questions based on the data from the [academia.stackexchange.com](https://academia.stackexchange.com/)
 datadump from [archive.org](https://archive.org/details/stackexchange)
 
 The project structure was setup using [pyscaffoldext-dsproject](https://github.com/pyscaffold/pyscaffoldext-dsproject).
 
## Technologies
Project is created with:
* setuptools >=38.3
* pyscaffold >=3.2a0, <3.3a0
	
## Setup
To run this project, install it locally using python:

```
$ python setup.py install
```

## Features
List of features ready and TODOs for future development
* [Initital data analysis](notebooks/1.0-me-initial-data-exploration.ipynb)
* [Document representation](notebooks/2.0-me-document-representation.ipynb)
* [Dimensionality reduction](notebooks/3.0-me-dimensionality-reduction.ipynb)
* [Classifier creation](notebooks/4.0-me-classification.ipynb)

To-do list:
* Classifier evaluation
* Final classifier ensamble -> Tag recommendation

## Status
Project is: _in progress_, _finished_, _no longer continue_ and why?

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
