import pandas as pd

#import dataset
movies = pd.read_csv("only_useful_datasets.csv")

"""function to add columns quarter and year"""
def years_quarters (dataframe):
    #change format of column "release_date" to DateTime
    movies.release_date = pd.to_datetime(movies.release_date)

    #assign new columns with 1) year and 2) quarter
    movies_with_year = movies.assign(year = movies.release_date.dt.year)
    movies_with_year_quarter = movies_with_year.assign(quarter = movies.release_date.dt.quarter)
    
    #drop release date (optional)
    #movies_with_year_quarter = movies_with_year_quarter.drop("release_date", 1)
    
    return movies_with_year_quarter


df = years_quarters(movies)

#Export
#df.to_csv("out.csv")

"""Option to reorder columns"""
#reordering columns is OPTIONAL! If not first function, then fcks it all up
#movies_with_year_quarter_reorder = movies_with_year_quarter[['Unnamed: 0', 'id', 'original_title', 'adult', 'budget', 'genres', 'revenue', 'release_date', 'year', 'quarter', 'belongs_to_collection', 'production_countries', 'production_companies', 'runtime']]

#print(movies_with_year_quarter_reorder.tail())

"""function to one-hot-encode column quarter"""
def encode_quarter(dataframe):
    
    return df


"""
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
"""

