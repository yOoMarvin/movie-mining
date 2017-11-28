import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/raw/movies_metadata.csv')
df.head()

df.describe()

# Mean budget and revenue
df['budget'].mean()
df['revenue'].mean()

# Mean productivity rate
print(df['revenue'].mean() / df['budget'].mean())

# Plotting budget and revenue
df['revenue'].plot(kind='hist')
plt.show()


# Scatter Revenue / Budget
plt.scatter(df['budget'], df['revenue'])
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()
