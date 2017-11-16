import pandas as pd
import re

def encodeProductionCompany(df):
    """
    MultipleHotEncode column production_companies
    Json loads does not work here
    Regex to extract needed values
    Seperate them with | for MultipleHotEncoding
    """

    new_values = []
    indices = []
    for index, row in df.iterrows():
        new_value = row["production_companies"].replace("[", "").replace("]", "").replace("{\'name\': ", "").replace("}", "").replace("\'", "")
        new_value = re.sub(", id: \d+", "", new_value)
        new_value = new_value.replace(", ", "|")
        new_values.append(new_value)
        indices.append(index)
    new_values_encoded = pd.Series(new_values,index=indices).str.get_dummies()
    return new_values_encoded

def encodeProductionCompanyToOne(df):
    """
    OneHotEncode column production_compnaies
    Json loads does not work here
    Regex to extract first value
    """
    new_values = []
    for index, row in df.iterrows():
        new_value = row["production_companies"].replace("[", "").replace("]", "").replace("{\'name\': ", "").replace(
            "}", "").replace("\'", "")
        new_value = re.sub(", id: \d+", "", new_value)
        new_value = new_value.split(", ")[0]
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    return new_values_encoded
