"""
Created on Thu Nov 16 13:07:06 2017

@author: Marvin
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# Custom imports
# Do not import all classifiers, rather look for the best result and fit them manually




# Import data
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)
print('data imported')
# DataFrame containing label (!)
#df = pd.DataFrame(data)

# Target
target = df['productivity_binned']

# Drop unwanted columns
columns_to_drop = [
        "original_title",
        "productivity_binned",
        "id"
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
#print(target.values)
#lb = LabelEncoder()
#target = lb.fit_transform(target)
#print(target)
knn_estimator.fit(df, target)
print('done doing the fit')

# Predict with cross validation
knn_scores = cross_val_score(knn_estimator, df, target, cv=cv, scoring='f1_micro')


print("KNN F1 Score: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))







# do evaluations on them and print them
# plot some nice graphs
