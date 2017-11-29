# -*- coding: utf-8 -*-

import ClassifierTemplate as ct
import pandas as pd


data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

label_column = "productivity_binned_binary"

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,label_column, False)

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


# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
#df = df.iloc[df.nonzero()[0]]
print(df)
print(c.data.columns)

#c.splitData()
#c.upsampleTrainData()
#c.downsampleTrainData()

# get parameters for GridSearch
scorer = c.f1(average="macro") # use F1 score with macro averaging
estimator = c.tree() # get decisionTree estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    'criterion':['entropy'],
    'max_depth':[100],
    'min_samples_split' :[5],
    'class_weight': [None]
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
Best scores
=====NO IMPROVEMENTS=====
SCORES: {'belongs_to_collection': 0.54956165011044289, 'budget': 0.55617722244541223, 'year': 0.53521848287812301, 'actor_': 0.57076429207634582, 'director_': 0.57170149590809127, 'company_': 0.56815009127848326, 'country_': 0.54834096585825476, 'genre_': 0.55564796394746707}
CURRENT: 0.5778687225361315, MAX: 0.5717014959080913, FEATURE: director_
DROPPED: ['quarter_', 'runtime', 'adult']
=========================
"""