import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#import dataset here and onehotencode here

"""
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)

#to integers
l_encoder = LabelEncoder()
values_encoded = l_encoder.fit_transform(values)
print(values_encoded)

#one-hot-encoding (binary)
binary_encoder = OneHotEncoder(sparse=False)#return array if sparse = false, otherwise returns matrix

values_encoded = values_encoded.reshape(len(values_encoded), 1)#reshape data
onehot_encoded = binary_encoder.fit_transform(values_encoded)
print(onehot_encoded)
"""
