import pandas as pd
import matplotlib.pyplot as plt
import numpy

data = pd.read_csv('../../data/interim/only_useful_datasets.csv')
data.head()

data['productivity'] = data['revenue'] / data['budget']



data_filtered = data[['original_title', 'budget','revenue', 'release_date', 'productivity']]
data_filtered[ data_filtered['revenue']<100 ].sort_values(by='revenue', ascending=False)


data_filtered

data_filtered['budget'].describe()

plt.figure(figsize=(10,8))
data_filtered['revenue'].plot()
plt.show()



# Kill outliers
arr = data_filtered['productivity']

elements = numpy.array(arr)

mean = numpy.mean(elements, axis=0)
sd = numpy.std(elements, axis=0)

final_list = [x for x in arr if (x > mean - 0.5 * sd)]
final_list = [x for x in final_list if (x < mean + 0.5 * sd)]
final_list
len(final_list)
