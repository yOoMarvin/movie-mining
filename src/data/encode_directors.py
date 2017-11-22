import pandas as pd
import re
import encode_production_company as epc
import pandas as pd
import re
import encode_production_company as epc

def encodeDirectorsToOne(df, filter, threshold):
    """
    reads the first actor out of the credits.csv and onehotencodes it
    joins the encoded actors with the movie id again
    :param df: Credits DF (../../data/raw/credits.csv/credits.csv)
    :return: encodedActors + id
    """
    directors = []
    indices = []
    prefix = "director_"
    for index, row in df.iterrows():
        director = re.search("\'name\': \'\w+(-* *\w*)*\'", row['crew'])
        if not(director is None):
            director = director.group().replace("'name': ", "").replace("'", "")
            director = prefix + director
            directors.append(director)
            indices.append(index)
        else:
            directors.append("")
    directors_encoded = pd.get_dummies(directors)

    #actors_encoded['id'] = pd.Series(df['id'])
    if(filter):
        directors_encoded = epc.filterWithThreshold(directors_encoded, threshold)
    #actors_encoded = epc.addPrefixToColumn(new_values_encoded, "actors")
    return directors_encoded

def directorsForHistogram(df):
    directors = []
    indices = []
    prefix = "director_"
    for index, row in df.iterrows():
        director = re.search("\'name\': \'\w+(-* *\w*)*\'", row['crew'])
        if not (director is None):
            director = director.group().replace("'name': ", "").replace("'", "")
            director = prefix + director
            directors.append(director)
            indices.append(index)
        else:
            directors.append("")
    new_values_encoded = pd.DataFrame()
    new_values_encoded['directors'] = pd.Series(directors)

    return new_values_encoded