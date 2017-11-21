# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:43:26 2017

@author: Steff
"""

import pandas as pd
import numpy as np

movies = pd.read_csv("../../data/external/imdb_measures.csv", index_col=0)

# extract budgets
budget = movies["raw"]
budget = budget[budget.str.contains("Budget</h5>").fillna(False)]
budget = budget.str.replace(r".+?Budget</h5>\\n([$,0-9]+) \(.+",r"\1")
#budget = budget.str.replace(r"^(?!\$).+","")
#budget = budget.replace("",np.nan)

df_budget = pd.DataFrame(budget)
df_budget = df_budget.dropna()

# extract revenue
revenue = movies["raw"]
revenue = revenue[revenue.str.contains(r"\([Ww]orldwide\)").fillna(False)]
revenue = revenue.str.replace(r".*?(\$[0-9,]+)\s\([Ww]orldwide\).*",r"\1")
#revenue = revenue.str.replace(r"^(?!\$).+","")
#revenue = revenue.replace("",np.nan)

df_revenue = pd.DataFrame(revenue)
df_revenue = df_revenue.dropna()

# join and save file
joined = df_budget.join(df_revenue,how="inner",lsuffix="_budget",rsuffix="_revenue")
joined.to_csv("../../data/external/imdb_measures_extracted.csv", encoding="utf-8")