"""
Created on Thu Nov 20

@author: Marvin
"""
import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned")

# drop columns not needed for Classification
c.dropColumns([
        "original_title",
        "id.1"
])

# get parameters for GridSearch
scorer = c.f1(average="micro") # use F1 score with micro averaging
estimator = c.bayes() # get GaussianNB estimator

cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42


# parameters to iterate in GridSearch
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
        ,cv=cv
)

# print best result
c.gridSearchBestScore(gs)

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"naivebayes_results.csv")
