# Script for processing the exportet data from async fetch


import pandas as pd



















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
