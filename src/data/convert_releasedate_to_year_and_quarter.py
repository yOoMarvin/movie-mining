import pandas as pd

#import dataset
movies = pd.read_csv("only_useful_datasets.csv")

#function
def years_quarters (dataframe):
    print(dataframe)
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
