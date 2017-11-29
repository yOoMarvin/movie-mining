import pandas as pd
import re
import encode_production_company as epc


def encodeActorsToOne(df, filter, threshold):
    """
    reads the first actor out of the credits.csv and onehotencodes it
    joins the encoded actors with the movie id again
    :param df: Credits DF (../../data/raw/credits.csv/credits.csv)
    :return: encodedActors + id
    """
    actors = []
    indices = []
    prefix = "actor_"
    for index, row in df.iterrows():
        actor = re.search("\'name\': \'\w+(-* *\w*)*\'", row['cast'])
        if not (actor is None):
            actor = actor.group().replace("'name': ", "").replace("'", "")
            actor = prefix + actor
            actors.append(actor)
            indices.append(index)
        else:
            actors.append("")
            indices.append(index)
    actors = pd.Series(actors,index=indices)
    actors_encoded = pd.get_dummies(actors)

    #actors_encoded['id'] = pd.Series(df['id'])
    if (filter):
        print('before filter')
        print(list(actors_encoded))
        print(threshold)
        actors_encoded = epc.filterWithThreshold(actors_encoded, threshold)
        print('after filter')
        print(list(actors_encoded))
    # actors_encoded = epc.addPrefixToColumn(new_values_encoded, "actors")
    return actors_encoded


def actorsForHistogram(df):
    actors = []
    indices = []
    prefix = "actor_"
    for index, row in df.iterrows():
        actor = re.search("\'name\': \'\w+(-* *\w*)*\'", row['cast'])
        if not (actor is None):
            actor = actor.group().replace("'name': ", "").replace("'", "")
            actor = prefix + actor
            actors.append(actor)
            indices.append(index)
        else:
            actors.append("")
    new_values_encoded = pd.DataFrame()
    new_values_encoded['actors'] = pd.Series(actors)
    #new_values_encoded = pd.DataFrame(actors, columns=['test'])
    #new_values_encoded.rename(columns={0: 'log(gdp)'}, inplace=True)

    return  new_values_encoded
