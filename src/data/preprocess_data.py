# skript to perfom preprocessing on the data
import pandas as pd
import numpy as np
import interesting_colums as ic
import convert_collection as cc
import convert_releasedate as cr
import encode_genre as eg
import productivity as p
import encode_production_company as epc
import encode_quarter as eq
import enconde_production_country as ep_country
import normalize_column as nc
import encode_actors as ea

# read in raw csv files
metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)


#limit metadata to relevant columns and rows only
metadata = ic.interesting_columns(metadata)
print('limited to interesting columns')
actors = pd.read_csv("../../data/raw/credits.csv", index_col=2)
metadata = pd.merge(metadata, actors, left_index=True, right_index=True)

# metadata: convert collection to boolean 
metadata = cc.collection_to_boolean(metadata)
print('collection is converted to boolean')

# metadata: convert year + encode quarter 
metadata = cr.years_quarters(metadata)
metadata = eq.quarter_encoding(metadata)
print('year converted, quarter encoded')

# metadata: encode company. country, genre and attach to dataframe. This is not done by the method itself
metadata = pd.concat([metadata, ep_country.encodeProductionCountry(metadata)], axis=1)
print('encoded country')
metadata = pd.concat([metadata, eg.encodeGenre(metadata)], axis=1)
print('encoded genre')
metadata = pd.concat([metadata, p.productivity_column(metadata)], axis=1)
print('encoded productivity')
metadata = pd.concat([metadata, epc.encodeProductionCompany(metadata)], axis=1)
print('encoded company')

# metadata: normalize data here if necessary. Input: df and string, Output: completed dataframe with normalized column Example: runtime
metadata = nc.normalize_column_data(metadata, 'runtime')
metadata = nc.normalize_column_data(metadata, 'year')
print('data normalized')

#process actor column (returned)

actors_column_processed = ea.encodeActorsToOne(metadata);
actors_column_processed = actors_column_processed.reset_index()
actors_column_processed = actors_column_processed.set_index(metadata.index)

# metadata: merge again with metadata
metadata = pd.concat([metadata, actors], axis=1)

# metadata: drop irrelevant data
metadata = metadata.drop([
        'budget'
        ,'genres'
        ,'revenue'
        ,'release_date'
        ,'production_countries'
        ,'production_companies'
        ,'quarter'
        ,'productivity'
],1)
print('dropped irrelevant data')


#safe dataset to file, important: encode as UTF-8
metadata.to_csv("../../data/interim/only_useful_datasets.csv", encoding='utf-8')
print('new dataset should be saved, doublcheck in folder')

