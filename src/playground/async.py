import asyncio
from aiohttp import ClientSession
import pandas as pd


df = pd.read_csv('../../data/raw/movies_metadata.csv')
#remove NaN's from imdb_id column
df = df[pd.notnull(df['imdb_id'])]
#remove rows where imdb_id is 0
df = df[df.imdb_id != '0']

# Part of the df
part = df.head(50)


async def hello(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            response = await response.read()
            print(response)



loop = asyncio.get_event_loop()



#base url for request. Later: append the imdb_id
base_url = "https://theimdbapi.org/api/movie?movie_id="


tasks = []


for index, row in part.iterrows():
    imdb_id = row['imdb_id']
    # build url
    url = base_url + imdb_id

    task = asyncio.ensure_future(hello(url.format(index)))
    tasks.append(task)
loop.run_until_complete(asyncio.wait(tasks))
