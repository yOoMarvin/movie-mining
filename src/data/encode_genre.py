import pandas as pd
import re


def encodeGenre(df):
    """
    MultipleHotEncode column genre
    Json loads does not work here
    Regex to extract needed values
    Seperate them with | for MultipleHotEncoding
    """
    new_values = []
    indices = []
    for index, row in df.iterrows():
        new_value = re.sub("{(\'id\': \d+, )*\'name\': ", "", row["genres"])
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").replace(", ", "|")
        indices.append(index)
        new_values.append(new_value)
    new_values_encoded = pd.Series(new_values, index=indices).str.get_dummies()
    return new_values_encoded

def encodeGenreToOne(df):
    """
    OneHotEncode column genre
    Json loads does not work here
    Regex to extract first value
    """
    new_values = []
    for index, row in df.iterrows():
        new_value = re.sub("{(\'id\': \d+, )*\'name\': ", "", row["genres"])
        new_value = new_value.replace("[", "").replace("]", "").replace("\'", "").replace("}", "").split(", ")[0]
        new_values.append(new_value)
    new_values_encoded = pd.get_dummies(new_values)
    return new_values_encoded