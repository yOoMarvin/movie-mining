# movie-mining
A data mining project which will predict the success of future movies. This is a student project at the University of Mannheim. HW17 Master of Science, Business Informatics
---
## Generate new Dataset
In order to generate a new Dataset for later usage the following two scripts have to be executed.

1. Execute preprocess_data.py

### preprocess_data.py
A script to perform preprocessing on the raw data. It involves the following steps:

+ Important: set values for filter (boolean), and threshold values in the beginning of the script (adjust variables)
+ limit data to datasets with budget and revenue value
+ convert columns to more usefule values
+ encode data inside the columns
+ normalize column data (Input: df and string, Output: completed dataframe with normalized column)
+ calls script to perfom test/train split

Each function called in the script recieves a Data Frame Object (DFO) and returns either a fully modified DFO or the new calculated columns (inc. indexes)

preprocess_data saves the data to only_useful_datasets.csv (encoding='utf-8')
---
