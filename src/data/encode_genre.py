import pandas as pd
import re

def encodeGenre():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_value = re.sub("{(\'id\': \d+, )*\'name\': ", "", row["genres"])
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").replace(", ", "|")
        new_values.append(new_value)
        print(new_value)
    new_values_encoded = pd.Series(new_values).str.get_dummies()
    print(new_values_encoded.head())
    return new_values_encoded

def encodeGenreToOne():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_value = re.sub("{(\'id\': \d+, )*\'name\': ", "", row["genres"])
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").split(", ")[0]
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    print(new_values_encoded.head())
    return new_values_encoded
