import pandas as pd
from sklearn import preprocessing
import numpy as np


# this method removes the initial column and replaces it with the normalized column

def normalize_column_data(df, column_name):
    x = df[column_name].astype(np.float64)
    x = x.values.reshape(-1, 1)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df[column_name] = pd.DataFrame(x_scaled, index=df.index.values)
    return df
