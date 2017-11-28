import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/raw/movies_metadata.csv')
df.head()

df.describe()


# Plotting budget and revenue
df['revenue'].plot(kind='hist')
plt.show()
