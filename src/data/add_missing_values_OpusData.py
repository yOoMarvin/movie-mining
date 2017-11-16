# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:11:37 2017

@author: Steff
"""

# skript to perfom preprocessing on the data
import pandas as pd

# read csv file with pre-limited data
metadata = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)
print("-------- ",metadata.shape," ---------\n")
data = metadata[["original_title","revenue","budget","release_date"]]
data = data.query('(revenue == 0 | budget == 0)')
data = data[data["release_date"].notnull()]
data = data[data.release_date.str.startswith('20')]
print("-------- ",data.shape," ---------\n")

data2 = pd.read_csv("../../data/external/MovieData.csv")
data2["revenue"] = data2["movie_financial_summary_domestic_box_office"] + data2["movie_financial_summary_international_box_office"]
data2 = data2[["movie_display_name","revenue","movie_financial_summary_production_budget","production_date_year"]]
data2.rename(columns={'movie_display_name': 'original_title', 'movie_financial_summary_production_budget': 'budget'}, inplace=True)
print("-------- ",data2.shape," ---------\n")

joined = data.set_index('original_title').join(data2.set_index('original_title'), lsuffix='_caller', rsuffix='_other')
joined = joined[joined["revenue_other"].notnull()]

print(joined.head())
print("-------- ",joined.shape," ---------\n")

"""

    !!!! Join is based on movie name -> need to take year into account (several rows are wrongly joined)

# joined.to_csv("joined.csv")
"""