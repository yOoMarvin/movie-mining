# Python script for preprocessing
# introduces a new column to the movie metadata dataset
# Budget / Revenue ratio

import pandas as pd

data = pd.read_csv('../../data/interim/only_useful_datasets.csv')
data.head()

data['productivity'] = data['revenue'] / data['budget']
