import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/raw/movies_metadata.csv')
df.head()

df.describe()

#Count zero values of budget and revenue
zeros = df.query('revenue == 0 & budget == 0')
zeros
not_zero = df.query('revenue > 0 & budget > 0')
not_zero

relevant = df.query('revenue > 10000 & budget > 10000')
relevant

# Mean budget and revenue
df['budget'].mean()
df['revenue'].mean()

# Mean productivity rate
print(df['revenue'].mean() / df['budget'].mean())

# Plotting budget and revenue
df['revenue'].plot(kind='hist')
plt.show()


# Scatter Revenue / Budget
plt.figure(figsize=(5,5))

plt.scatter(relevant['budget'], relevant['revenue'])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='productivity 1', alpha=.8) # draw diagonal

plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.legend()
plt.show()

# Productivity
prd = not_zero['revenue'] / not_zero['budget']
prd
