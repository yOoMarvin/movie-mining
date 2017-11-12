import pandas as pd

#import dataset here and onehotencode here
"""function to one-hot-encode column quarter """
def quarter_encoding(movies):
    mv_enc_q = pd.get_dummies(movies.quarter, sparse=False)
    movies_new = pd.concat([movies, mv_enc_q], axis=1)
    return movies_new