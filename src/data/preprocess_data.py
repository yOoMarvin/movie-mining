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
import train_test_split as splitter
import encode_directors as ed
import adjust_measures as adj


# set values for the thresholding during preprocessing
filter = True
threshold_actors = 0.076
threshold_companies = 0.025
threshold_directors = 0.05

# read in raw csv files
metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)

metadata = adj.adjust_measures(metadata)

status = '[Status: ]'

#limit metadata to relevant columns and rows only
metadata = ic.interesting_columns(metadata)
print(status + 'limited to interesting columns')
actors = pd.read_csv("../../data/raw/credits.csv", index_col=2)
metadata = pd.merge(metadata, actors, left_index=True, right_index=True)

# metadata: convert collection to boolean
metadata = cc.collection_to_boolean(metadata)
print(status + 'collection is converted to boolean')

# metadata: convert year + encode quarter
metadata = cr.years_quarters(metadata)
metadata = eq.quarter_encoding(metadata)
print(status + 'year converted, quarter encoded')

# metadata: encode company. country, genre and attach to dataframe. This is not done by the method itself
metadata = pd.concat([metadata, ep_country.encodeProductionCountry(metadata)], axis=1)
print(status + 'encoded country')

metadata = pd.concat([metadata, eg.encodeGenre(metadata)], axis=1)
print(status + 'encoded genre')

metadata = pd.concat([metadata, p.productivity_column(metadata)], axis=1)
print(status + 'encoded productivity')

#print(epc.encodeProductionCompany(metadata))
metadata = pd.concat([metadata, epc.encodeProductionCompany(metadata, filter, threshold_companies)], axis=1)
#print(encodedCompanies.keys())
print(status + 'encoded company')

# metadata: normalize data here if necessary. Input: df and string, Output: completed dataframe with normalized column Example: runtime
metadata = nc.normalize_column_data(metadata, 'runtime')
print(status + 'data normalized')



# keep productivity in a seperate file
productivity = metadata[["productivity","productivity_binned"]]
productivity.to_csv("../../data/processed/productivity.csv", encoding='utf-8')
print(status + 'productivity safed in different file...done')


#process actor column (returned)
actors_column_processed = ea.encodeActorsToOne(metadata, filter, threshold_actors)
actors_column_processed = actors_column_processed.reset_index()
actors_column_processed = actors_column_processed.set_index(metadata.index)
#print(actors_column_processed.keys())
print(status + 'encoded actors')

# preprocess directors_column
directors_column_processed = ed.encodeDirectorsToOne(metadata, filter, threshold_directors)
directors_column_processed = directors_column_processed.reset_index()
directors_column_processed = directors_column_processed.set_index(metadata.index)
print(status + 'encoded directors')

# metadata: merge again with metadata
metadata = pd.concat([metadata, actors_column_processed], axis=1)
metadata = pd.concat([metadata, directors_column_processed], axis=1)

# metadata: drop irrelevant data
metadata = metadata.drop([
        'budget'
        ,'genres'
        ,'revenue'
        ,'release_date'
        ,'production_countries'
        ,'production_companies'
        ,'quarter'
        ,'year' # dropped year because it will cause unseen data for future values
        ,'productivity'
        ,'cast' # not needed anymore after preprocessing
        ,'crew'
],1)
print(status + 'dropped irrelevant data')
print(metadata.head())
#safe dataset to file, important: encode as UTF-8
metadata.to_csv("../../data/interim/only_useful_datasets.csv", encoding='utf-8')

print('new dataset should be safed, doublcheck in folder')

# execute train-test-split
splitter.split_dataset()
