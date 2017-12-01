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

print(len(c.data.columns))
thrCompany = 2
thrActor =2
thrDirector=2
print("thresholds: company: {}, actor: {}, director: {}".format(thrCompany, thrActor, thrDirector))
c.thresholdByColumn(thrCompany,"company")
c.thresholdByColumn(thrActor,"actor")
c.thresholdByColumn(thrDirector,"director")
print(len(c.data.columns))

### scale something if needed
c.scale([
        "budget"
])

### drop columns by prefix if needed
#c.dropColumnByPrefix("actor")
#c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
#c.dropColumnByPrefix("country")
#c.dropColumnByPrefix("genre")
#c.dropColumnByPrefix("quarter_")


# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
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
    'max_depth':[None],
    'min_samples_split' :[3],
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

with new feature filter
=====NO IMPROVEMENTS=====
SCORES: {'adult': 0.56832355630840548, 'belongs_to_collection': 0.54552982242115577, 'budget': 0.56533610221930708, 'year': 0.55302453796530227, 'director_': 0.56897703666807009, 'country_': 0.56257342284862588, 'genre_': 0.56027504231726943, 'quarter_': 0.57073402912833682}
CURRENT: 0.5768253837458162, MAX: 0.5707340291283368, FEATURE: quarter_
DROPPED: ['actor_', 'company_', 'runtime']
========================= actor 8 rest 3

=====NO IMPROVEMENTS=====
SCORES: {'adult': 0.55713436143526462, 'belongs_to_collection': 0.56388874332639138, 'budget': 0.55288825037831435, 'year': 0.55652709310402149, 'actor_': 0.56921430005829299, 'director_': 0.55997121038889008, 'company_': 0.54672604119589152, 'country_': 0.55427642787528264, 'genre_': 0.56046682979848339, 'quarter_': 0.55895251402911361}
CURRENT: 0.5706414908501328, MAX: 0.569214300058293, FEATURE: actor_
DROPPED: ['runtime']
========================= filter: actor10, rest 5, no scale of budget

=====NO IMPROVEMENTS=====
SCORES: {'adult': 0.56230982648563421, 'belongs_to_collection': 0.56307661699906142, 'budget': 0.55710101019643066, 'runtime': 0.56098224862446944, 'year': 0.54351015026207283, 'actor_': 0.56902838267469635, 'director_': 0.56195302604225994, 'company_': 0.5476013416936959, 'genre_': 0.56240353824055012}
CURRENT: 0.5754642520870411, MAX: 0.5690283826746964, FEATURE: actor_
DROPPED: ['country_', 'quarter_']
=========================filter actor 11 rest 4
"""