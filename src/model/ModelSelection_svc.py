# -*- coding: utf-8 -*-

import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned_binary")

# drop columns not needed for Classification
c.dropColumns([
         "original_title"
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

## scale something if needed
c.scale([
        "budget"
])

### drop columns by prefix if needed
c.dropColumnByPrefix("actor_")
#c.dropColumnByPrefix("director_")
#c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter_")

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
print(df)
print(c.data.columns)

# get parameters for GridSearch
scorer = c.f1(average="macro") # use F1 score with micro averaging
estimator = c.svc() # get svc estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    'multi_class':['ovr'],
    'class_weight':[None, 'balanced']
    #'class_weight': [{'yes':1, 'no':5}, {'yes':1, 'no':3}, {'yes':1, 'no':1}, None]
    # parameter can be used to tweak parallel computation / n = # of jobs
    #,"n_jobs":[1]
}

"""
# compute GridSearch
gs = c.gridSearch(
        estimator
        ,scorer
        ,parameters
        ,print_results=False # let verbose print the results
        ,verbose=2
        ,cv=cv
        ,onTrainSet=False
)

# print best result
c.gridSearchBestScore(gs)

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"svc_results.csv")
"""

# calculate cross validation: try samplings
estimator.set_params(
    multi_class='ovr',
    class_weight='balanced',
    random_state=42
)

print(c.cross_validate(cv,estimator,sample=""))
c.plot_coefficients(estimator=estimator,top_features=20)


"""

 with new filters:
  --------------------------- GRID SEARCH BEST SCORE ---------------------------
 Best score is 0.6004784738065798 with params {'class_weight': 'balanced', 'multi_class': 'ovr'}.
 ------------------------------------------------------------------------------
 DROPPED: ['quarter_', 'adult', 'actor_', 'director_']
 thresholds: actor:8, company:3, director:3
 'no': 1464, 'yes': 2452
 
 
 dropping almost everything best score:
 =====NO IMPROVEMENTS=====
SCORES: {'belongs_to_collection': 0.59311269404023259, 'budget': 0.60132955023024537, 'runtime': 0.60328404840386463, 'year': 0.59440458095313331, 'director_': 0.60037294719307421, 'company_': 0.56556106734817957}
CURRENT: 0.6069228601464032, MAX: 0.6032840484038646, FEATURE: runtime
DROPPED: ['actor_', 'country_', 'quarter_', 'genre_', 'adult']
=========================
"""

