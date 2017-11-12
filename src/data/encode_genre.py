import pandas as pd
import re

def encodeGenre():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        one_genre = re.sub("{(\'id\': \d+, )*\'name\': ", "", row["genres"])
        one_genre = one_genre.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").split(", ")[0]
        new_values.append(one_genre)
    new_values_encoded = pd.get_dummies(new_values)
    print(new_values_encoded.head())
    return new_values_encoded

encodeGenre();