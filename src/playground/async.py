import asyncio
from aiohttp import ClientSession
import pandas as pd


# Try out something new:
# Read the dataset and export a list with the urls needed to fetch
# Fetch them all and write every json response in a file
# In another script process that file

df = pd.read_csv('../../data/raw/movies_metadata.csv')
#remove NaN's from imdb_id column
df = df[pd.notnull(df['imdb_id'])]
#remove rows where imdb_id is 0
df = df[df.imdb_id != '0']


#base url for request. Later: append the imdb_id
base_url = "https://theimdbapi.org/api/movie?movie_id="

#initialize the url list
urls = []

for index, row in df.iterrows():
    urls.append(base_url + row['imdb_id'])
