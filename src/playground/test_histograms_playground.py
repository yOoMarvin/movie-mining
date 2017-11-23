import matplotlib.pyplot as plt
import pandas as pd
import normalize_column as nc

metadata = pd.read_csv("../../data/interim/only_useful_datasets.csv")


metadata.diff().hist(alpha=0.5, bins=50)

#print(metadata.head())
#metadata = nc.normalize_column_data(metadata, 'productivity')
#metadata['productivity'].plot(kind='hist')

#plt.show()
#print(metadata.head())