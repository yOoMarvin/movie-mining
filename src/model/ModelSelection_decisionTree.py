# -*- coding: utf-8 -*-

import ClassifierTemplate as ct
import pandas as pd


data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned_binary", True)

# drop columns not needed for Classification
c.dropColumns([
        "original_title"
        #,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        ,"year"
        #,"quarter"
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
c.dropColumnByPrefix("quarter_")


# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
#df = df.iloc[df.nonzero()[0]]
print(df)
print(c.data.columns)


# get parameters for GridSearch
scorer = c.f1(average="micro") # use F1 score with micro averaging
estimator = c.tree() # get decisionTree estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    'criterion':['gini', 'entropy'],
    'max_depth':[1, 2, 3, 4, 5, 10, 50, 100, None],
    'min_samples_split' :[2,3,4,5],
    'class_weight': [{'yes':1, 'no':1}]
    # parameter can be used to tweak parallel computation / n = # of jobs
    #,"n_jobs":[1]
}


# compute GridSearch
gs = c.gridSearch(
        estimator
        ,scorer
        ,parameters
        ,print_results=False # let verbose print the results
        ,verbose=2
        ,cv=cv
)

# print best result
c.gridSearchBestScore(gs)

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"tree_results.csv")

c.splitData()

c.fit_predict(gs.best_estimator_)
print(c.confusion_matrix())

c.classification_report()