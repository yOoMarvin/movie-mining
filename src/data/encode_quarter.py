import pandas as pd
from addPrefixToColumn import addPrefixToColumn

#import dataset here and onehotencode here
"""function to one-hot-encode column quarter """
def quarter_encoding(movies):
    mv_enc_q = pd.get_dummies(movies.quarter, sparse=False)
    mv_enc_q = addPrefixToColumn(mv_enc_q, "quarter")
    movies_new = pd.concat([movies, mv_enc_q], axis=1)
    return movies_new