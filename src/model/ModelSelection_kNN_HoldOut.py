# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:09:06 2017

@author: Steff
"""

import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
c.dropColumns([
        "original_title"
        #,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        #,"year"
        #,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

### scale something if needed
#c.scale([
#        "budget"
#])

### drop columns by prefix if needed
c.dropColumnByPrefix("actor")
c.dropColumnByPrefix("director")
c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter_")

# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
print(df)
print(c.data.columns)

# get information about the data
c.balanceInfo()

estimator = c.knn() # get estimator

c.splitData()
c.fit_predict(estimator)

c.classification_report()