import pandas as pd

#import dataset
dataframe = pd.read_csv("only_useful_datasets.csv")

"""function to add columns quarter and year"""
def years_quarters (movies):
    #change format of column "release_date" to DateTime
    movies.release_date = pd.to_datetime(movies.release_date)

    #assign new columns with 1) year and 2) quarter
    movies_with_year = movies.assign(year = movies.release_date.dt.year)
    movies_with_year_quarter = movies_with_year.assign(quarter = movies.release_date.dt.quarter)
    
    #drop release date (optional)
    #movies_with_year_quarter = movies_with_year_quarter.drop("release_date", 1)
    
    return movies_with_year_quarter


df = years_quarters(dataframe)

#Export
#df.to_csv("out.csv")

