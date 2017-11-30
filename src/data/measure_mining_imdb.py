# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:46:39 2017

@author: Steff
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlopen
import re
import time
import sys

#sys.stdout = open("measure_mining_out.txt", "w")

#os.chdir("D:\Master\Data Mining HWS2017\movie-mining\src\data")

filepath_raw = "../../data/raw/"
filepath_interim = "../../data/external/"
filename = "imdb_measures.csv"
split_distance = 1000
iterations = 0
total = 0

def init():
    movies = pd.read_csv(filepath_raw + "movies_metadata.csv", index_col=5)
    # filtered = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)
    #data = pd.DataFrame(movies.loc[filtered.index.values]["imdb_id"])
    data = pd.DataFrame(movies["imdb_id"])
    print(data.head())
    data["budget"] = np.nan
    data["revenue"] = np.nan
    data["raw"] = np.nan
    data = data.drop_duplicates()
    data.to_csv("../../data/interim/" + filename, encoding='utf-8')
    
def load():
    global total
    global iterations
    data = pd.read_csv(filepath_interim + filename, index_col=0)
    total = len(data)
    iterations = 0
    return data

def split(data):
    dist = split_distance
    edge = dist
    splits = []
    while edge < len(data)-dist:
        splits.append(pd.DataFrame(data.iloc[edge-dist:edge]))
        edge += dist
    splits.append(pd.DataFrame(data.iloc[edge-dist:len(data)]))
    return splits

def iterate(data,splitted):
    global iterations
    global total
    savepoint = 100
    #i = 0
    for movie_id, imdb_id in data["imdb_id"].iteritems():
        print("iterate:",movie_id)
        #i += 1
        iterations += 1
        if pd.isnull(data.loc[movie_id]["raw"]) and not pd.isnull(imdb_id):
            print("------------\n{} {} sending request".format( movie_id,imdb_id ))
            html = urlopen("http://www.imdb.com/title/" + imdb_id + "/business").read()
            match = re.search(r"(<div id=\"tn15content\">.+<table>)",str(html))
            data.loc[movie_id,"raw"] = match.group(0)
            print("{}/{} done. {} till checkpoint.\n------------".format( iterations,total,savepoint ))
            
            savepoint -= 1
            if (savepoint < 0):
                print("\n------------\nsavepoint reached - saving ...\n------------\n")
                savepoint = 100
                save(splitted)
            #break
        else:
            print(movie_id,imdb_id,"not sending request")
    print("\n------------\nend of iteration - saving ...\n------------\n")
    save(splitted)
        

def save(splitted):
    data = pd.concat(splitted)
    data.to_csv(filepath_interim + filename, encoding='utf-8')
    
def start():
    #init()
    data = load()
    splitted = split(data)    
    for df in splitted:
        iterate(df,splitted)
        
start()
    

"""
# extend missing
data = load()
movies = pd.read_csv(filepath_raw + "movies_metadata.csv", index_col=5)
# filtered = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)
#data = pd.DataFrame(movies.loc[filtered.index.values]["imdb_id"])
movies = pd.DataFrame(movies["imdb_id"])
movies["budget"] = np.nan
movies["revenue"] = np.nan
movies["raw"] = np.nan
movies = movies.drop_duplicates()

print(len(movies))
movies = movies.drop(data.index.values)
print(len(movies))
print(len(data))
data = pd.concat([movies,data])
print(len(data))
data.to_csv(filepath_interim + filename, encoding='utf-8')
"""