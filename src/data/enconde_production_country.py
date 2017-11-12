import pandas as pd;
import re;

def encodePoductionCountry():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    new_values = []
    for index, row in movies.iterrows():
        new_value = re.sub("{\'iso_\d+_\d+\': ", "", row["production_countries"])
        new_value = re.sub(", \'name\': \'(\w+ *)*\'", "", new_value)
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").replace(", ", "|")
        new_values.append(new_value)
    new_values_encoded = pd.Series(new_values).str.get_dummies()
    print(new_values_encoded.head())
    return new_values_encoded

def encodePoductionCountryToOne():
    movies = pd.read_csv("../../data/interim/only_useful_datasets.csv")
    print(movies["production_countries"].head())
    new_values = []
    for index, row in movies.iterrows():
        new_value = re.sub("{\'iso_\d+_\d+\': ", "", row["production_countries"])
        new_value = re.sub(", \'name\': \'(\w+ *)*\'", "", new_value)
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").split(", ")[0]
        print(new_value)
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    return new_values_encoded