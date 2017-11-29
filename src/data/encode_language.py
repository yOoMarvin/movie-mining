import pandas as pd
from addPrefixToColumn import addPrefixToColumn

#import dataset here and onehotencode here
"""function to one-hot-encode column quarter """
def language_encoding(movies):
    mv_enc_l = pd.get_dummies(movies['original_language'], sparse=False)
    mv_enc_l = addPrefixToColumn(mv_enc_l, "language")
    languages_new = pd.concat([movies, mv_enc_l], axis=1)
    return languages_new