# academia_tag_recommender

Tag Recommender for [academia.stackexchange.com](https://academia.stackexchange.com/)

## Table of contents
* [Description](#description)
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
* [Classifier evaluation](notebooks/5.0-me-evaluation.ipynb)
* [Problem transformation](notebooks/6.0-me-problem-transformation.ipynb)
* [Algorithm adaption](notebooks/7.0-me-algorithm-adaption.ipynb)
* [Classifier ensembles](notebooks/8.0-me-ensemble.ipynb)
* [Classification using Word2Vec Embedding](notebooks/9.0-me-classification-word2vec.ipynb) _in progress_
* [Classification using Doc2Vec Embedding](notebooks/9.1-me-classification-doc2vec.ipynb) _in progress_
* [Classification using Fasttext Embedding](notebooks/9.2-me-classification-fasttext.ipynb) _in progress_
* [Classification using classwise undersampling](notebooks/10.0-me-undersampling.ipynb)

To-Do:
* Embedding _in progress_
* Tag recommendation

## Status
Project is: _in progress_

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
