# Python script for preprocessing
# introduces a new column to the movie metadata dataset
# Budget / Revenue ratio

import pandas as pd
"""


data = pd.read_csv('../../data/interim/only_useful_datasets.csv')
data.head()

data['productivity'] = data['revenue'] / data['budget']

# Playground: binning
data_binned = pd.DataFrame(dict(
    productivity = pd.cut(data['productivity'], bins=3, labels=['low', 'middle', 'high'])
))
data_binned_and_encoded = pd.get_dummies(data_binned)
data_binned_and_encoded
"""







## Function for adding the new column
def productivity_rating_bins(df):
    df['productivity'] = df['revenue'] / df['budget']
    # Binning and one hot encoding
    df_binned = pd.DataFrame(dict(
        productivity_binned = pd.cut(
                df['productivity']
                # bins from 0-1, 0-2, 2-5, 5-open
                ,bins=[0.0,1.0,2.0,5.0,float("inf")] # If bins is a sequence it defines the bin edges allowing for non-uniform bin width
                ,right=False # Indicates whether the bins include the rightmost edge or not
                ,labels=['unproductive', 'smallProductivity', 'goodProductivity', 'highProductivity']
                ,include_lowest=True # Whether the first interval should be left-inclusive or not.
        )
    ),index=df.index.values)
    #df_binned_and_encoded = pd.get_dummies(df_binned)
    #return df_binned_and_encoded
    return df_binned







## Function for adding the new column with just two kind of bins
## Bins only in yes and no
def productivity_binary_bins(df):
    df['productivity'] = df['revenue'] / df['budget']
    # Binning and one hot encoding
    df_binned = pd.DataFrame(dict(
        productivity_binned = pd.cut(
                df['productivity']
                # bins only for productive and unproductive
                ,bins=[0.0,1.0,float("inf")] # If bins is a sequence it defines the bin edges allowing for non-uniform bin width
                ,right=False # Indicates whether the bins include the rightmost edge or not
                ,labels=['no', 'yes']
                ,include_lowest=True # Whether the first interval should be left-inclusive or not.
        )
    ),index=df.index.values)
    #df_binned_and_encoded = pd.get_dummies(df_binned)
    #return df_binned_and_encoded
    return df_binned
