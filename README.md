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
1. [Initital data analysis](notebooks/1.0-me-initial-data-exploration.ipynb)
2. [Document representation](notebooks/2.0-me-document-representation.ipynb)
    1. [Bag of words](notebooks/2.1-me-bag-of-words.ipynb)
    2. [Embedding](notebooks/2.2-embedding.ipynb)
    3. [Dimensionality reduction](notebooks/2.3-me-dimensionality-reduction.ipynb)
3. [Evaluation metrics](notebooks/3.0-me-evaluation-metrics.ipynb)
4. [Classification](notebooks/4.0-me-classification.ipynb) _in progress_
    1. Classification using Bag Of Words
        1. [Classification using Tfidf](notebooks/4.1.0-me-classification-bow.ipynb)
        2. [Classification using Count](notebooks/4.2.1-me-classification-count.ipynb)
    2. [Classification using Word2Vec Embedding](notebooks/4.2-me-classification-word2vec.ipynb)
    3. [Classification using Doc2Vec Embedding](notebooks/4.3-me-classification-doc2vec.ipynb)
    4. [Classification using Fasttext Embedding](notebooks/4.4-me-classification-fasttext.ipynb)
    5. Classwise Classification
        1. [Basic Classwise Classification](notebooks/4.5.0-me-classwise.ipynb)
        2. [Classwise Classification using classwise gridsearch](notebooks/4.5.1-me-gridsearch.ipynb)
        3. [Classwise Classification using classwise undersampling](notebooks/4.5.2-me-classwise-undersampling.ipynb)
        4. [Classwise Classification using classwise undersampling and gridsearch](notebooks/4.5.3-me-undersampling-gridsearch.ipynb)
        5. [Classwise Classification using classwise multiple](notebooks/4.5.4-me-classwise-multiple.ipynb)
5. [Performance comparison](notebooks/5.0-me-performance-comparison.ipynb) _in progress_
6. [Recommendation](notebooks/6.0-me-recommender.ipynb) _in progress_

## Status
Project is: _in progress_

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
