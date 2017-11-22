import pandas as pd
import matplotlib.pyplot as plt
import numpy

prd = pd.read_csv('../../data/processed/productivity.csv', index_col=0)
prd.head()

df = pd.read_csv('../../data/interim/only_useful_datasets.csv')
df['productivity_binned']


from collections import Counter
counts = Counter(prd['productivity_binned'])
df = pd.DataFrame.from_dict(counts, orient='index')
df.plot(kind='bar')
plt.show()

counts
prd.sort_values(by='productivity', ascending=False)
