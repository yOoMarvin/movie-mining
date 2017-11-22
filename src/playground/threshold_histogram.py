import pandas as pd
import encode_actors as ac
import interesting_colums as ic
import adjust_measures as adj
import matplotlib.pyplot as plt
import encode_production_company as epc
import encode_directors as ed

metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)

metadata = adj.adjust_measures(metadata)

status = '[Status: ]'

#limit metadata to relevant columns and rows only
metadata = ic.interesting_columns(metadata)
print(status + 'limited to interesting columns')
actors = pd.read_csv("../../data/raw/credits.csv", index_col=2)
metadata = pd.merge(metadata, actors, left_index=True, right_index=True)

values_actors = ac.actorsForHistogram(metadata)
#values_actors['actors'].value_counts().plot(kind='bar')
#plt.show()

values_company = epc.companiesForHistogramm(metadata)
#values_company['companies'].value_counts().plot(kind='bar')
#plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)

values_directors = ed.directorsForHistogram(metadata)
values_directors['directors'].value_counts().plot(kind='bar')

plt.suptitle("Histogramm Distribution")
plt.show()
size = len(metadata)
print(size)