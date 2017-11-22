import pandas as pd
import matplotlib.pyplot as plt
import numpy

# productivity values
df = pd.read_csv('../../data/processed/productivity.csv')
df[df['Unnamed: 0'] == 1362]

# Look for values that are binned wrong
for index, row in df.iterrows():
    if type(row['productivity_binned']) == float:
        print(str(row['Unnamed: 0']) + ' ' + str(type(row['productivity_binned'])))


# Which movie is the bad one?
movies = pd.read_csv('../../data/interim/only_useful_datasets.csv')
movies.head()

mv = movies[['Unnamed: 0', 'original_title', 'productivity_binned']]
mv[mv['Unnamed: 0'] == 1362]

# Check the value in the measures
measures = pd.read_csv('../../data/external/measures.csv')
measures.head()
er = measures[measures['id'] == 1362]
er
# print the types and prd from those values
for index, row in er.iterrows():
    print('Budget type: ' + str(type(row['budget'])))
    print('Revenue type: ' + str(type(row['revenue'])))
    print(row['revenue'] / row['budget'])

# What were the values in raw?
raw = pd.read_csv('../../data/raw/movies_metadata.csv')
hobbit = raw[raw['id'] == 1362]
hobbit[['revenue', 'budget']]
