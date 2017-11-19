# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:02:17 2017

@author: Steff
"""

from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

data_train, data_test = train_test_split(
    data
    ,test_size=0.3
    ,random_state=42
    ,stratify=data["productivity_binned"]
)

print("Train size: {}, Test size: {} ".format(len(data_train),len(data_test)))

data_train.to_csv("../../data/processed/train_set.csv", encoding='utf-8')
data_test.to_csv("../../data/processed/test_set.csv", encoding='utf-8')

print('datasets saved')