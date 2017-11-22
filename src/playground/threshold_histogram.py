import pandas as pd
import encode_actors as ac
import interesting_colums as ic
import adjust_measures as adj
import matplotlib.pyplot as plt

metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)

metadata = adj.adjust_measures(metadata)

status = '[Status: ]'

#limit metadata to relevant columns and rows only
metadata = ic.interesting_columns(metadata)
print(status + 'limited to interesting columns')
actors = pd.read_csv("../../data/raw/credits.csv", index_col=2)
metadata = pd.merge(metadata, actors, left_index=True, right_index=True)
new_values = ac.actorsForHistogram(metadata)
print(new_values.head())
new_values['test'].value_counts().plot(kind='bar')
plt.show()
size = len(metadata)
print(size)