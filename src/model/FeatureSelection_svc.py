# -*- coding: utf-8 -*-

import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

label_column = "productivity_binned_binary"

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,label_column)

# drop columns not needed for Classification
c.dropColumns([
         "original_title"
        #,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

### scale something if needed
#c.scale([
#        "budget"
#])

### drop columns by prefix if needed
#c.dropColumnByPrefix("actor")
#c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
#c.dropColumnByPrefix("country")
#c.dropColumnByPrefix("genre")
#c.dropColumnByPrefix("quarter_")

print(len(c.data.columns))
thrCompany = 3
thrActor =8
thrDirector=3
print("thresholds: company: {}, actor: {}, director: {}".format(thrCompany, thrActor, thrDirector))
c.thresholdByColumn(thrCompany,"company")
c.thresholdByColumn(thrActor,"actor")
c.thresholdByColumn(thrDirector,"director")
print(len(c.data.columns))

# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
#print(df)
print(c.data.columns)



# get parameters for GridSearch
scorer = c.f1(average="macro") # use F1 score with macro averaging
estimator = c.svc() # get svc estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    'multi_class':['ovr'],
    'class_weight':['balanced']
    #'class_weight': [{'yes':1, 'no':5}, {'yes':1, 'no':3}, {'yes':1, 'no':1}, None]
    # parameter can be used to tweak parallel computation / n = # of jobs
    #,"n_jobs":[1]
}


features = [
            "adult",
            "belongs_to_collection",
            "budget",
            "runtime",
            "year",
            "actor_",
            "director_",
            "company_",
            "country_",
            "genre_",
            "quarter_"
]

# compute FeatureSelect
gs = c.featureselect_greedy(
        features
        ,parameters
        ,scorer
        ,estimator
        ,cv
        ,label_column
)

"""
=====NO IMPROVEMENTS=====
SCORES: {'belongs_to_collection': 0.5562500378516172, 'budget': 0.57859298966949679, 'runtime': 0.57331267034189004, 'year': 0.56026353327085598, 'company_': 0.56227028114319433, 'country_': 0.5632187472306216, 'genre_': 0.57230307165306782}
CURRENT: 0.5786578739249424, MAX: 0.5785929896694968, FEATURE: budget
DROPPED: ['quarter_', 'adult', 'actor_', 'director_']
=========================
"""

