# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:35:51 2017

@author: Steff
"""

import pandas as pd
import json as j

data = pd.read_csv("../../data/raw/movies_metadata.csv", index_col=5)

#print(len(data))
#print(data.loc[43139])

comp = data["production_companies"]
comp = comp.str.replace("  "," ")
comp = comp.str.replace(" , ",", ")
comp = comp.str.replace(': \"',": \'")
comp = comp.str.replace(', \"',", \'")
comp = comp.str.replace('",',"',")
comp = comp.str.replace("\"","'")
comp = comp.str.replace("'name': '","\"name\": \"")
comp = comp.str.replace("', 'id':","\", \"id\":")
comp = comp.str.replace("'id':","\"id\":")
comp = comp.str.replace("\\","")
comp = comp.str.replace("xa0","")
#comp = comp.str.replace("'name':","\"name\":").replace("'id':","\"name\":")

  
def parseCompany(m_id, c_id, c_name):
    #print(m_id,c_id,c_name)
    data_company_to_movie.append({
            "company": c_id,
            "movie": m_id
    })
    data_companies[c_id] = c_name
    if (c_id == 0):
        print(m_id,c_name)
    
data_company_to_movie = []
data_companies = {}

i = 0
for index, row in comp.iteritems():
    i += 1
    
    #print(row)
    try:
        jrow = j.loads(row)
        for c in jrow:
            parseCompany(index,c["id"],c["name"])
    except ValueError:
        print("\n----------------\n",index,row,"\n----------------\n")
    except TypeError:
        print("\n----------------\n",index,row,"\n----------------\n")
        
    #if (i>3):
    #    break

c2m_df = pd.DataFrame(data_company_to_movie)
c_df = pd.DataFrame(
        list(data_companies.values())
        ,index=list(data_companies.keys())
        ,columns=["name"]
)
#c_df.set_index("id",inplace=True)

c2m_df.to_csv("../../data/interim/companies_to_movies.csv", encoding='utf-8')
c_df.to_csv("../../data/interim/companies.csv", encoding='utf-8')