import pandas as pd

data = pd.read_csv('../../data/interim/only_useful_datasets.csv')
data.head()

data['productivity'] = data['revenue'] / data['budget']



data_filtered = data[['original_title', 'budget','revenue', 'release_date', 'productivity']]
data_filtered[ data_filtered['budget']>500000 ].sort_values(by='productivity', ascending=False)
