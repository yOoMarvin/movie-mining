# Mining the Success for Movies
A data mining project which will predict the success of future movies. This is a student project at the University of Mannheim. HW17 Master of Science, Business Informatics

---

## Application Area and Goals
### Problem Statement
Before new movies are being produced, every stakeholder is interested in the monetary success of the intended movie. In order to predict the success, costly methods are being applied, such as market investigations or analyses. The benefit of Data Mining to the analysis of large datasets can also be transferred to the stated problem of predicting a movie’s success.

### Goals
The goal of this project is to learn a model which will predict how successful a not yet released movie will be. This is done by using common data mining techniques in the Python programming language using the machine learning models provided by the library _scikit-learn_. As the main objective the question ”_Based on revenue, will the movie be popular or will it be a flop?_” shall be answered for all possible combinations of information on a new movie as precisely as possible.

## Datasets
The selected dataset onto which a classification model shall be learned is provided by _Kaggle_. It is named [_The Movies Dataset_](https://www.kaggle.com/rounakbanik/the-movies-dataset) and contains metadata of approximately 45,000 movies in its raw format. It is provided and updated by Rounak Banik. The complete dataset consists of several files in _csv-format_ containing data about movie casts, metadata and external scores. The main file used during preprocessing is named _movies-metadata.csv_.
**Note:** not all datasets are provided in this repository due to large file sizes. Additionally, the _.zip_ folder in the directory _data/raw/_ must be extracted to ensure the functionality of the preprocessing scripts.

## Generate new Dataset
In order to generate a new Dataset for later usage the script _preprocess-data.py_ must be executed.

Afterwards you get new csv file in _data/processed/_ named _train-set.csv_. This file can be used for further data mining and classification.

## Feature and Model Selection
In order to be more flexible with selected features and parameters a classifier template was introduced. With the help of this it is possible to search for the features which will result in the best performance. Hyper parameter tuning is done via GridSearch from the _scikit-learn_ Framework. Scripts can be found under _src/model_.
