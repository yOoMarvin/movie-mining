import pandas as pd
import re
from addPrefixToColumn import addPrefixToColumn

def encodeProductionCountry(df):
    """
    MultipleHotEncode column production_countries
    Json loads does not work here
    Regex to extract needed values
    Seperate them with | for MultipleHotEncoding
    """
    new_values = []
    indices = []
    for index, row in df.iterrows():
        new_value = re.sub("{\'iso_\d+_\d+\': ", "", row["production_countries"])
        new_value = re.sub(", \'name\': \'(\w+ *)*\'", "", new_value)
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").replace(", ", "|")
        indices.append(index)
        new_values.append(new_value)
    new_values_encoded = pd.Series(new_values, index=indices).str.get_dummies()
    new_values_encoded = addPrefixToColumn(new_values_encoded, "country")
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
    new_values_encoded = addPrefixToColumn(new_values_encoded, "country")
    return new_values_encoded
