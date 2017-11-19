# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:45:22 2017

@author: Steff
"""


import pandas as pd

data = pd.read_csv("../../data/interim/companies.csv", index_col=0)

comp = data["name"]
comp = comp.str.replace("[^(?u)\w\s]+"," ") # remove anything not alphanumeric or whitespace
comp = comp.str.lower()
comp = comp.str.replace(r"\s+(entertainment|productions|films|film|pictures|company|gmbh|kg)$","") # remove keywords at the end of the sentence if not only word
comp = comp.str.replace(r"\s\(.+\)$","") # remove anything ending in brackets
comp = comp.str.replace(r"(\w+\s+)(\w+\s+)(\w+)(\s+\w+)+",r"\1\2\3") # if its longer than x words, cut it
comp = comp.str.replace("(([a-z]+\s+)+)(\d+)$",r"\1") # if it starts with a word but ends with a number, remove the number
comp = comp.str.replace("\s{2,}"," ") # remove all double whitespaces
comp = comp.str.replace("^\s+","") # remove all leading whitespaces
comp = comp.str.replace("\s+$","") # remove all ending whitespaces

df = pd.DataFrame(comp)
df.to_csv("../../data/interim/companies_cleaned.csv", encoding='utf-8')