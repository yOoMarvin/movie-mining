import asyncio
from aiohttp import ClientSession
import pandas as pd
import json


# Try out something new:
# Read the dataset and export a list with the urls needed to fetch
# Fetch them all and write every json response in a file
# In another script process that file

df = pd.read_csv('../../../data/raw/movies_metadata.csv')
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






# async handling

async def fetch(url, session):
    async with ClientSession() as session:
        async with session.get(url) as response:
            print(url)
            # Todo: Try to just return the needed values!
            result = await response.json()
            return result

async def bound_fetch(sem, url, session):
    # Getter function with semaphore.
    async with sem:
        await fetch(url, session)

# Run function, create a list of tasks, gather all responses at the end
async def run(r):
    # global variable for access the results later
    global imdb
    tasks = []
    sem = asyncio.Semaphore(1000)

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
future = asyncio.ensure_future(run(1000))
loop.run_until_complete(future)

# Debug print
#print(imdb)

# Writing the list into a file
# Important: You have to add the [] to make sure the file is an array
# Important: Also add the , for separating the the objects
# Important: The .strip('"') kills the "" that are written in the file
f = open('output.json', 'w')
f.write("[".strip('"'))

for item in imdb:
    json.dump(item, f, separators=(',', ':'))
    # adding a comma to the end of each json object
    f.write(','.strip('"'))
else:
    # Don't append the "," for the last item in the list!
    json.dump(item, f, separators=(',', ':'))

f.write("]".strip('"'))
f.close()
