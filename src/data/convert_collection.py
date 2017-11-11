# loops over all collection data, converts to true if collection - false if not part of collection

import pandas as pd

interesting_data = pd.read_csv("../../data/interim/only_useful_datasets.csv")


interesting_data.belongs_to_collection = interesting_data.belongs_to_collection.astype(str)


interesting_data['belongs_to_collection'] = (interesting_data['belongs_to_collection'] != 'nan').astype(int)

interesting_data.belongs_to_collection = interesting_data.belongs_to_collection.astype(bool)
print(interesting_data.belongs_to_collection)