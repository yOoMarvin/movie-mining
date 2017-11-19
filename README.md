# movie-mining
A data mining project which will predict the success of future movies. This is a student project at the University of Mannheim. HW17 Master of Science, Business Informatics
---
## Generate new Dataset
In order to generate a new Dataset for later usage the following two scripts have to be executed.

1. Execute preprocess_data.py
2. Execute train_test_split.py

### preprocess_data.py
A script to perform preprocessing on the raw data. It involves the following steps:

+ limit data to datasets with budget and revenue value
+ convert columns to more usefule values
+ encode data inside the columns
+ normalize column data (Input: df and string, Output: completed dataframe with normalized column)

Each function called in the script recieves a Data Frame Object (DFO) and returns either a fully modified DFO or the new calculated columns (inc. indexes)

preprocess_data saves the data to only_useful_datasets.csv (encoding='utf-8')

### train_test_split.py
A script to read in the only_useful_datasets.csv and to split it into train and test set.

+ Read in CSV file
+ Set params
+ save data to train_set.csv and test_set.csv
---
