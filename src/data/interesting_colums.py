# import
import pandas as pd
import numpy as np
from sklearn import preprocessing


def interesting_columns(metadata):
    # convert budget from object to float variabel
    metadata.budget = metadata.budget.astype(np.float64)
    # focus on columns needed.
    metadata = metadata.loc[:,
               ['id', 'original_title', 'adult', 'budget', 'genres', 'revenue', 'release_date', 'belongs_to_collection',
                'production_countries', 'production_companies', 'runtime']]
    # query dataset based on valid revenue and valid budget, save to csv file
    metadata = metadata.query('revenue > 100000 & budget > 100000 & genres != "[]" & production_companies != "[]"')
    return metadata
