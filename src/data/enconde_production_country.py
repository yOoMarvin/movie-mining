import pandas as pd
import re

def encodePoductionCountry(df):
    """
    MultipleHotEncode column production_countries
    Json loads does not work here
    Regex to extract needed values
    Seperate them with | for MultipleHotEncoding
    """
    new_values = []
    for index, row in df.iterrows():
        new_value = re.sub("{\'iso_\d+_\d+\': ", "", row["production_countries"])
        new_value = re.sub(", \'name\': \'(\w+ *)*\'", "", new_value)
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").replace(", ", "|")
        new_values.append(new_value)
    new_values_encoded = pd.Series(new_values).str.get_dummies()
    return new_values_encoded

def encodePoductionCountryToOne(df):
    """
    OneHotEncode column production_countries
    Json loads does not work here
    Regex to extract first value
    """
    new_values = []
    for index, row in df.iterrows():
        new_value = re.sub("{\'iso_\d+_\d+\': ", "", row["production_countries"])
        new_value = re.sub(", \'name\': \'(\w+ *)*\'", "", new_value)
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").split(", ")[0]
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    return new_values_encoded
