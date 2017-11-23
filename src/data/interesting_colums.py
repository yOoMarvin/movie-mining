# import
import pandas as pd
import numpy as np


def interesting_columns(metadata):
    # convert budget from object to float variabel
    metadata.budget = metadata.budget.astype(np.float64)
    # focus on columns needed.
    metadata = metadata.loc[:,
               ['original_title', 'adult', 'budget', 'genres', 'revenue', 'release_date', 'belongs_to_collection',
                'production_countries', 'production_companies', 'runtime']]
    # query dataset based on valid revenue and valid budget, save to csv file
    
    # threshold for revenue and budget not needed anymore
    # metadata = metadata.query('revenue > 100000 & budget > 100000 & genres != "[]" & production_companies != "[]"')
    metadata = metadata.query('genres != "[]" & production_companies != "[]"')
    
    print("deleting duplicates. before: ",len(metadata))
    metadata = metadata.drop_duplicates(keep="first")
    print("deleting duplicates. after: ",len(metadata))
    
    return metadata
