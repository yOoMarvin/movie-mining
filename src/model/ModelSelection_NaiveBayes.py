"""
Created on Thu Nov 20

@author: Marvin
"""
import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned_binary", upsample=True)

### drop single columns not needed for Classification
c.dropColumns([
        "original_title"
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
        #,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

### scale something if needed
#c.scale([
#        "budget"
#])

### drop columns by prefix if needed
"""
c.dropColumnByPrefix("actor")
c.dropColumnByPrefix("director")
c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter_")
"""
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("actor")
c.dropColumnByPrefix("director")

# lets print all non-zero columns of a movie to doublecheck
#df = c.data.loc[19898]
#df = df.iloc[df.nonzero()[0]]
#print(df)
#print(c.data.columns)



# get information about the data
c.balanceInfo()



# get parameters for GridSearch
scorer = c.f1(average="macro") # use F1 score with micro averaging
estimator = c.bayes() # get GaussianNB estimator

cross_val = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42


parameters = {
    # No parameters for bayes

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
        ,cv=cross_val
)

# print best result
c.gridSearchBestScore(gs)

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"naivebayes_results.csv")

# variables for export
estimator = gs.best_estimator_
data = c.data
target = c.truth_arr