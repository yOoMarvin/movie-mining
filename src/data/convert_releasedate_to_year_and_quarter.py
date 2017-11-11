#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:01:40 2017

@author: Dan
"""
import pandas as pd
#import dataset
movies = pd.read_csv("only_useful_datasets.csv")

#change format of column "release_date" to DateTime
movies.release_date = pd.to_datetime(movies.release_date)

#assign new columns with 1) year and 2) quarter
movies_with_year = movies.assign(year = movies.release_date.dt.year)
movies_with_year_quarter = movies_with_year.assign(quarter = movies.release_date.dt.quarter)

"""
#reoder is OPTIONAL! If not first function, then fcks it all up
#reorder columns
movies_with_year_quarter_reorder = movies_with_year_quarter[['Unnamed: 0', 'id', 'original_title', 'adult', 'budget', 'genres', 'revenue', 'release_date', 'year', 'quarter', 'belongs_to_collection', 'production_countries', 'production_companies', 'runtime']]

#print(movies_with_year_quarter_reorder.tail())
"""

#Export
movies_with_year_quarter.to_csv("out_reorder.csv")

"""
def functionname( parameters ):
   "function_docstring"
   function_suite
   return [expression]
"""