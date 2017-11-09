##import data
import pandas as pd
import numpy as np
metadata = pd.read_csv("../../data/raw/movies_metadata.csv")

##convert budget to float variable, print results
metadata.budget = metadata.budget.astype(np.float64)
#print (metadata.budget)

##metadata.loc[:, 'adult':'revenue']
print(metadata.keys())

#focus on columns needed.
metadata = metadata.loc[:, [ 'id', 'original_title', 'adult', 'budget','genres','revenue', 'release_date', 'belongs_to_collection', 'production_countries' , 'production_companies', 'runtime'   ]]
print(metadata.keys())

#query dataset based on valid revenue and valid budget

#hasRevenue = metadata['revenue'] != 0
#hasBudget  = metadata['budget'] != 0

metadata = metadata.query('revenue > 0 & budget > 0')
metadata.to_csv("../../data/interim/only_useful_datasets.csv")

#print(metadata.keys())