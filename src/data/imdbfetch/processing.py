# Script for processing the exportet data from async fetch


import pandas as pd
import json
from pprint import pprint



with open('output.json', encoding='utf-8') as data_file:
   data = json.loads(data_file.read())

pprint(data)


















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
