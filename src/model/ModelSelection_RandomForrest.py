import numpy as np

from time import time
from scipy.stats import randint as sp_randint
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import dropColumnByPrefix as dcbp
import ClassifierTemplate as ct

# get some data
data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)
# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df, "productivity_binned")

c.dropColumns([
    "original_title"
    # ,   "adult"
])

scorer = c.f1(average="micro")  # use F1 score with micro averaging
estimator = c.randomForrest()  # get decisionTree estimator
cv = c.fold(
    k=10
    , random_state=42
)  # KStratifiedFold with random_state = 42

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
start = time()
gs = c.gridSearch(
    estimator
    , scorer
    , param_grid
    , print_results=False  # let verbose print the results
    , verbose=2
    , cv=cv
)

# run grid search


start = time()
# print best result
c.gridSearchBestScore(gs)
print("GridSearchCV took %.2f"
      % (time() - start))
# save all results in csv
c.gridSearchResults2CSV(gs,param_grid,"tree_results.csv")
