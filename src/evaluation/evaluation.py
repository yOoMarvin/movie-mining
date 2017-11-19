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


# import all classifiers
# do evaluations on them
# plot some nice graphs
