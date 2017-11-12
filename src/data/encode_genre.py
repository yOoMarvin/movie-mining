import pandas as pd
import json
import re

def encodeGenre():
    #movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    movies = pd.read_csv("../../data/raw/movies_metadata.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_row = row["genres"].replace("[", "").replace("]", "").replace("'", "\"")
        newer = re.sub("{\"id\": \d+, \"name\": ", "", new_row)
        newer = newer.replace("\"", "").replace("}", "").split(", ")[0]
        new_values.append(newer)
        #print(newer)
    movies["genres"] = new_values
    print(movies.head())

encodeGenre();