import pandas as pd
import re

def encodeProductionCompany():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_value = row["production_companies"].replace("[", "").replace("]", "").replace("{\'name\': ", "").replace("}", "").replace("\'", "")
        new_value = re.sub(", id: \d+", "", new_value)
        new_value = new_value.replace(", ", "|")
        new_values.append(new_value)
    new_values_encoded = pd.Series(new_values).str.get_dummies()
    return new_values_encoded

def encodeProductionCompanyToOne():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_value = row["production_companies"].replace("[", "").replace("]", "").replace("{\'name\': ", "").replace(
            "}", "").replace("\'", "")
        new_value = re.sub(", id: \d+", "", new_value)
        new_value = new_value.split(", ")[0]
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    return new_values_encoded
