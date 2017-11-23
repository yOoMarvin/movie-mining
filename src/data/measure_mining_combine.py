# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:41:05 2017

@author: Steff
"""
import pandas as pd

filepath_raw = "../../data/raw/"
filepath_external = "../../data/external/"

movies = pd.read_csv(filepath_raw + "movies_metadata.csv", index_col=5)
date = movies["release_date"]
date = date[date.str.contains(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").fillna(False)]
date = date.str.slice(0,4)
year = pd.DataFrame(date)

imdb = pd.read_csv(filepath_external + "imdb_measures_extracted.csv", index_col=0)
numbers = pd.read_csv(filepath_external + "thenumbers_measures_extracted.csv", index_col=0)

# drop all in numbers which is already present in imdb
numbers = numbers.drop(numbers.join(imdb,how="inner",rsuffix="_").index.values)

numbers = numbers.join(year,how="left")
numbers["raw"] = numbers["raw"].astype(int)
numbers["release_date"] = numbers["release_date"].astype(int)
numbers["diff"] = abs(numbers["raw"]-numbers["release_date"])
numbers = numbers[numbers["diff"]<3]

combined = pd.concat([
        imdb,
        numbers[["raw_budget","raw_revenue"]]
])

combined.columns = ["budget","revenue"]

combined.to_csv(filepath_external + "measures.csv", encoding="utf-8")