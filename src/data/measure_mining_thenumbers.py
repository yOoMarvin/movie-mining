# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:46:39 2017

@author: Steff
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlopen
import urllib.error
import re
import time
import sys

#sys.stdout = open("measure_mining_out.txt", "w")

#os.chdir("D:\Master\Data Mining HWS2017\movie-mining\src\data")

filepath_raw = "../../data/raw/"
filepath_interim = "../../data/external/"
#filepath_raw = ""
#filepath_interim = ""
filename = "thenumbers_measures.csv"
split_distance = 1000
iterations = 0
total = 0
savepoint_dist = 20

def init():
    print("loading",filepath_raw + "movies_metadata.csv")
    movies = pd.read_csv(filepath_raw + "movies_metadata.csv", index_col=5)
    titles = movies["title"].str.replace(r"[ +]","-")
    titles = titles.str.replace(r"[\.,'\":!?#$&%/*@’]","")
    titles = titles.str.replace(r"-+","-")
    titles = titles.str.replace("é","e")
    #print(titles[titles.str.contains(r"[^0-9a-zA-z-]").fillna(False)])
    print(len(titles))
    titles = titles.drop(titles[titles.str.contains(r"[^0-9a-zA-z-]").fillna(False)].index.values)
    print(len(titles))
    data = pd.DataFrame(titles)
    data["budget"] = np.nan
    data["revenue"] = np.nan
    data["raw"] = np.nan
    data = data.drop_duplicates()
    data.to_csv(filepath_interim + filename, encoding='utf-8')
    
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
    global savepoint_dist
    savepoint = savepoint_dist
    #i = 0
    for movie_id, title in data["title"].iteritems():
        print("iterate:",movie_id)
        #i += 1
        iterations += 1
        if pd.isnull(data.loc[movie_id]["raw"]) and not pd.isnull(title):
            try:
                print("------------\n{} {} sending request".format( movie_id,title ))
                savepoint -= 1
                html = urlopen("http://www.the-numbers.com/movie/" + title).read()
                #match = re.search(r"(<div id=\"tn15content\">.+<table>)",str(html))
                data.loc[movie_id,"raw"] = str(html) #match.group(0)
                print("{}/{} done. {} till checkpoint.\n------------".format( iterations,total,savepoint ))
            except urllib.error.HTTPError as e:
               if e.code == 404:
                   print("MOVIE NOT FOUND. GOING NEXT.",e)
                   data.loc[movie_id,"raw"] = "error occurred. movie not found."
               else:
                    print("ERROR OCCURRED. GOING NEXT.",e)
                    data.loc[movie_id,"raw"] = "error occurred."
                    time.sleep(5)
                    
            if (savepoint < 0):
                print("\n------------\nsavepoint reached - saving ...\n------------\n")
                savepoint = savepoint_dist
                save(splitted)
        else:
            print(movie_id,title,"not sending request")
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
        
#try:
start()
#except Exception as e:
#    print("!!! CRASH !!! WAITING AND RESTARTING !!!",e)
#    time.sleep(30)
#    start()