# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 01:19:25 2017

@author: Steff
"""

import pandas as pd

def adjust_measures(metadata):
    measures = pd.read_csv("../../data/external/measures.csv", index_col=0)
    
    # remove everything else
    metadata = metadata.loc[measures.index.values]
    # remove duplicates
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    measures = measures[~measures.index.duplicated(keep='first')]
    
    # assign new budget values    
    metadata = measures.combine_first(metadata)
    
    return metadata