import pandas as pd
import encode_actors as ac
import interesting_colums as ic
import adjust_measures as adj
import matplotlib.pyplot as plt
import encode_production_company as epc
import encode_directors as ed
import numpy as np

metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)

metadata = adj.adjust_measures(metadata)

status = '[Status: ]'

#limit metadata to relevant columns and rows only

metadata.budget = metadata.budget.astype(np.float64)
# focus on columns needed.
metadata = metadata.loc[:,
           ['original_title', 'adult', 'budget', 'genres', 'revenue', 'release_date', 'belongs_to_collection',
            'production_countries', 'production_companies', 'runtime', 'original_language']]

# query dataset based on valid revenue and valid budget, save to csv file

# threshold for revenue and budget not needed anymore
# metadata = metadata.query('revenue > 100000 & budget > 100000 & genres != "[]" & production_companies != "[]"')
metadata = metadata.query('genres != "[]" & production_companies != "[]"')

print("deleting duplicates. before: ", len(metadata))
metadata = metadata.drop_duplicates(keep="first")
print("deleting duplicates. after: ", len(metadata))


print(status + 'limited to interesting columns')
actors = pd.read_csv("../../data/raw/credits.csv", index_col=2)
metadata = pd.merge(metadata, actors, left_index=True, right_index=True)

values_actors = ac.actorsForHistogram(metadata)
#values_actors['actors'].value_counts().plot(kind='bar')
#plt.show()

values_company = epc.companiesForHistogramm(metadata)
values_company['companies'].value_counts().plot(kind='bar')
#plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)

values_directors = ed.directorsForHistogram(metadata)
#values_directors['directors'].value_counts().plot(kind='bar')

plt.suptitle("Histogramm Distribution")
plt.show()
size = len(metadata)
print(size)