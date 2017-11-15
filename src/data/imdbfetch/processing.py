# Script for processing the exportet data from async fetch


import pandas as pd
import json
from pprint import pprint

# Create empty dataframe for the results
result = pd.DataFrame(columns=['imdb_id', 'revenue', 'budget'])


# Open file and read from it
with open('output.json', encoding='utf-8') as data_file:
   data = json.loads(data_file.read())


# Iterate over idx and item of the list
# Fetch the data
for i, item in enumerate(data):
    revenue = process_json(item['metadata']['gross'])
    budget = process_json(item['metadata']['budget'])
    imdb_id = item['imdb_id']

    # If revenue and budget are not 0 append it to the result df
    if revenue != 0 and budget != 0:
        # Creating a DataFrame to append
        result = result.append(pd.DataFrame({'imdb_id': imdb_id,
                                            'revenue': revenue,
                                            'budget': budget},
                                             index=[i]))

result.head()
















def process_json(json):
    # split  into array because of additional information
    json_processed = json.split(' ')[0]
    # only extract digits from processed json
    result = ''.join([i for i in json_processed if i.isdigit()])

    #try to cast result to float, if fails, return 0 because there is no value from imdb
    try:
        return float(result)
    except ValueError:
        return 0
