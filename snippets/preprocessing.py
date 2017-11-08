
# ENCODING
from category_encoders.ordinal import OrdinalEncoder# choose one encoder
encoder = OrdinalEncoder()

#from category_encoders.one_hot import OneHotEncoder
#encoder = OneHotEncoder()

golf_encoded = encoder.fit_transform(golf[['Outlook', 'Temperature', 'Humidity', 'Wind']])
print(golf_encoded.head())


# BINNING
## EQUAL WIDTH
import pandas as pd
items = [0,4,12,16,16,18,24,26,28]
pd.cut(items, bins=3)

## EQUAL FREQUENCY
pd.qcut(items, q=3)

## APPLY TO DATASET
iris = pd.read_csv("iris.csv")
pd.cut(iris['SepalLength'], bins=3, labels=['low', 'middle', 'high'])

## CREATE BINNED DATASET
iris_binned = pd.DataFrame(dict(
    SepalLength = pd.cut(iris['SepalLength'], bins=3, labels=['low', 'middle', 'high']),
    SepalWidth = pd.cut(iris['SepalWidth'], bins=3, labels=['low', 'middle', 'high'])
))
iris_binned.head()

## GET DUMMIES
iris_binned_and_encoded = pd.get_dummies(iris_binned)
iris_binned_and_encoded.head()
