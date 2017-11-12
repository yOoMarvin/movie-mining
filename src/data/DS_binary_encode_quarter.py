import pandas as pd

#import dataset here and onehotencode here
"""function to one-hot-encode column quarter """
def quarter_encoding(movies):
    #pd.get_dummies(movies.quarter, sparse=False)
    mv_enc_q = pd.get_dummies(movies.quarter, sparse=False)
    
    movies_new = pd.concat([movies, mv_enc_q], axis=1)
    
    return movies_new

    

df = pd.read_csv("out.csv")
print(quarter_encoding(df))


#Export
#quarter_encoding(df).to_csv("out2.csv")
