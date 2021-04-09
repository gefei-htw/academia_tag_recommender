# academia_tag_recommender

Tag Generator for [academia.stackexchange.com](https://academia.stackexchange.com/) based on the data dump from July 9th 2020.


## Table of contents
* [Description](#description)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)

## Description

This project aims to generate tags for questions based on the [academia.stackexchange.com](https://academia.stackexchange.com/)
 datadump from [archive.org](https://archive.org/details/stackexchange) uploaded July 9th 2020.
It was developed as part of my masterthesis. The results can be obtained from the notebooks described in (#features).
The trained transformers and classifiers are not included in the published project since their file size is far too big.
Running all experiments resulted in 199 different variations of classifiers and an overall project size of 106 GB.

 The project structure was setup using [pyscaffoldext-dsproject](https://github.com/pyscaffold/pyscaffoldext-dsproject).
 
## Technologies
Project is created with:
* setuptools >=38.3
* pyscaffold >=3.2a0, <3.3a0
	
## Setup
To run this project, install it locally using python:

```
$ python setup.py develop
```

To conveniently work with the jupyter notebooks it is recommended to install anaconda and use jupyter lab.

## Features
List of features ready
1.	[Initital data analysis](notebooks/1.0-me-initial-data-exploration.ipynb)
2.	[Document representation](notebooks/2.0-me-document-representation.ipynb)
    1.	[Bag of words](notebooks/2.1-me-bag-of-words.ipynb)
    2.	[Embedding](notebooks/2.2-me-embedding.ipynb)
    3.	[Dimensionality reduction](notebooks/2.3-me-dimensionality-reduction.ipynb)
3.	[Evaluation metrics](notebooks/3.0-me-evaluation-metrics.ipynb)
4.	[Classification](notebooks/4.0-me-classification.ipynb)
    1.	[Tfidf](notebooks/4.1-me-classification-bow.ipynb)
    2.	[Count](notebooks/4.2-me-classification-count.ipynb)
    3.	[Word2Vec](notebooks/4.3-me-classification-word2vec.ipynb)
    4.	[Doc2Vec](notebooks/4.4-me-classification-doc2vec.ipynb)
    5.	[Fasttext](notebooks/4.5-me-classification-fasttext.ipynb)
    6.	[Basic Classwise Classification](notebooks/4.6.0-me-classwise.ipynb)
        1.	[Classwise Classification using gridsearch](notebooks/4.6.1-me-classwise-gridsearch.ipynb)
        2.	[Classwise Classification using undersampling](notebooks/4.6.2-me-classwise-undersampling.ipynb)
        3.	[Classwise Classification using undersampling and gridsearch](notebooks/4.6.3-me-classwise-undersampling-gridsearch.ipynb)
        4.	[Classwise Classification using multiple base classifiers](notebooks/4.6.4-me-classwise-multiple.ipynb)
5.	[Performance comparison](notebooks/5.0-me-performance-comparison.ipynb)
6.	[Tag Generator](notebooks/6.0-me-generator.ipynb)



## Status
Project is: _Inactive_

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
