import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/raw/movies_metadata.csv')
df

df.describe()

#Count zero values of budget and revenue
zeros = df.query('revenue == 0 & budget == 0')
zeros
not_zero = df.query('revenue > 0 & budget > 0')
not_zero.sort_values(by='revenue')

relevant = df.query('revenue > 10000 & budget > 10000')
relevant

# Mean budget and revenue
relevant['budget'].mean()
relevant['revenue'].mean()

# Mean productivity rate
print(relevant['revenue'].mean() / relevant['budget'].mean())

# Plotting budget and revenue
df['revenue'].plot(kind='hist')
plt.show()


# Scatter Revenue / Budget
f, ax = plt.subplots(figsize=(10, 10))
relevantSample = relevant.sample(2000) # This is the importante line
plt.scatter(relevantSample['budget'], relevantSample['revenue'])
plt.xlabel("Budget in $ 100 Mio.")
plt.ylabel("Revenue in $ 100 Mio-")
plt.xlim(0, 500000000)
plt.ylim(0, 500000000)
plt.gca().set_aspect('equal', adjustable='box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="r")

plt.show()

# Productivity
prd = not_zero['revenue'] / not_zero['budget']
prd


#Most common genre
genres = []
for item in df['genres']:
    parts = item.split(',')
    if len(parts) > 3:
        part = parts[3]
    genre = part[10:-2]
    genre = genre.replace("'", "")
    genres.append(genre)


print('Fantasy: ' + str(genres.count('Fantasy')))
print('Drama: ' + str(genres.count('Drama')))
print('Science Fiction: ' + str(genres.count('Science Fiction')))
print('Comedy: ' + str(genres.count('Comedy')))
print('Documentary: ' + str(genres.count('Documentary')))
print('Horror: ' + str(genres.count('Horror')))
print('Animation: ' + str(genres.count('Animation')))
print('Thriller: ' + str(genres.count('Thriller')))
print('Adventure: ' + str(genres.count('Adventure')))
print('Action: ' + str(genres.count('Action')))
print('Romance: ' + str(genres.count('Romance')))


# Scatter plot Revenue / Genre
plt.figure(figsize=(5,5))
plt.scatter(genres, relevant['budget'])
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()


#Scatter plot Revenue / Runtime
plt.figure(figsize=(10,10))
relevantSample = relevant.sample(500) # This is the importante line
plt.scatter(relevantSample['runtime'], relevantSample['revenue'])
plt.xlabel("Runtime")
plt.ylabel("Revenue")
plt.ylim(0, 500000000)
plt.show()

# Avg. number of actors in movie
cr = pd.read_csv('../../data/raw/credits.csv')
cast = cr['cast']
sampleCast = cast.sample(1000) # This is the importante line
