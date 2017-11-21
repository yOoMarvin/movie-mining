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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# Custom imports
# Do not import all classifiers, rather look for the best result and fit them manually
import ClassifierTemplate as ct



# Import data
data = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Target
target = df['productivity_binned']

# Drop unwanted columns
columns_to_drop = [
        "original_title",
        "id.1"
]
df = df.drop(columns_to_drop, axis=1)



# KStratifiedFold with random_state = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



##### KNN #####

print('#########   KNN EVALUATION   #########')

knn_estimator = KNeighborsClassifier(n_neighbors=3,
                            algorithm="auto",
                            weights="uniform",
                            p=2,
                            metric="manhattan")

#ERROR HERE
knn_estimator.fit(df, target)

# Predict with cross validation
knn_scores = cross_val_score(knn_estimator, df, target, cv=cv, scoring='f1')


print("KNN F1 Score: %0.2f (+/- %0.2f)" % (knn_scores.mean(), scores.std() * 2))







# do evaluations on them and print them
# plot some nice graphs
