import pandas as pd
import matplotlib.pyplot as plt
import numpy

prd = pd.read_csv('../../data/processed/productivity.csv')
prd.head()


from collections import Counter
counts = Counter(prd['productivity_binned'])
df = pd.DataFrame.from_dict(counts, orient='index')
df.plot(kind='bar')
plt.show()

counts
prd.sort_values(by='productivity', ascending=False)
