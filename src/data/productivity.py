# Python script for preprocessing
# introduces a new column to the movie metadata dataset
# Budget / Revenue ratio

import pandas as pd

data = pd.read_csv('../../data/interim/only_useful_datasets.csv')
data.head()

data['productivity'] = data['revenue'] / data['budget']

# Playground: binning
data_binned = pd.DataFrame(dict(
    productivity = pd.cut(data['productivity'], bins=3, labels=['low', 'middle', 'high'])
))
data_binned_and_encoded = pd.get_dummies(data_binned)
data_binned_and_encoded








## Function for adding the new column
def productivity_column(df):
    df['productivity'] = df['revenue'] / df['budget']
    # Binning and one hot encoding
    df_binned = pd.DataFrame(dict(
        productivity = pd.cut(df['productivity'], bins=3, labels=['low', 'middle', 'high'])
    ))
    df_binned_and_encoded = pd.get_dummies(df_binned)
    return df_binned_and_encoded
