# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:53:04 2017

@author: Steff
"""

from urllib.request import urlopen
html = urlopen("http://www.imdb.com/title/tt1772230/business").read()
print(html)