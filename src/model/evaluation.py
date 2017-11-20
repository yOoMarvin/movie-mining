"""
Created on Thu Nov 16 13:07:06 2017

@author: Marvin
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import NearestCentroid

from sklearn.model_selection import StratifiedKFold


# Custom imports
# Do not import all classifiers, rather look for the best result and fit them manually
import ClassifierTemplate as ct



# Import data
data_train = pd.read_csv("../../data/processed/train_set.csv", index_col=0)
data_test = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# Targets
target_train = df_train['productivity_binned']
target_test = df_test['productivity_binned']

# Drop unwanted columns
columns_to_drop = [
        "original_title",
        "id.1"
]
data_train = data_train.drop(columns_to_drop,axis=1)
data_test = data_test.drop(columns_to_drop, axis=1)

# KStratifiedFold with random_state = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)







##### KNN #####

print('#########   KNN EVALUATION   #########')

knn_estimator = KNeighborsClassifier(n_neighbors=3,
                            algorithm="auto",
                            weights="uniform",
                            p=2,
                            metric="manhattan")

# Error here... TO DO!
knn = knn_estimator.fit(df_train, target_train)
knn_predict = knn_estimator.predict(df_test)

# Print report
classification_report(target_test,knn_predict)










# do evaluations on them and print them
# plot some nice graphs
