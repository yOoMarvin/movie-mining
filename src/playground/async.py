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
base_url = "http://theimdbapi.org/api/movie?movie_id="

#initialize the url list
urls = []

# fill the url list
for index, row in df.iterrows():
    urls.append(base_url + row['imdb_id'])

urls






# async handling

async def fetch(url, session):
    async with ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            return result


# Run function, create a list of tasks, gather all responses at the end
async def run(r):
    # global variable for access the results later
    global imdb
    tasks = []

    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession() as session:
        for i in range(r):
            task = asyncio.ensure_future(fetch(urls[i], session))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        # you now have all response bodies in this variable
        # set the global variable to the value of the responses
        # results in a list of jsons
        imdb = responses


# Running the async call
loop = asyncio.get_event_loop()
future = asyncio.ensure_future(run(10))
loop.run_until_complete(future)

# Debug print
print(imdb)
