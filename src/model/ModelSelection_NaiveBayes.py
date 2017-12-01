"""
Created on Thu Nov 20

@author: Marvin
"""
import ClassifierTemplate as ct
import pandas as pd

df = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
#df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
c.dropColumns([
        "original_title"
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])


### drop columns by prefix if needed
c.dropColumnByPrefix("actor")
c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
#c.dropColumnByPrefix("quarter_")

#Threshold
#c.thresholdByColumn(1,"company")

"""

# get parameters
scorer = c.f1(average="macro") # use F1 score with micro averaging
estimator = c.bayes() # get GaussianNB estimator

cross_val = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42


print("unsampled:" + str(c.cross_validate(cross_val,estimator,sample="")))
print("upsampled:" + str(c.cross_validate(cross_val,estimator,sample="up")))
print("downsampled:" + str(c.cross_validate(cross_val,estimator,sample="down")))



BEST: downsampled: 0.54
"""
