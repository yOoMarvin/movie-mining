# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:44 2017

@author: Dan
"""

import pandas as pd
import ClassifierTemplate as ct
from time import time
import datetime

def RandomForest(df, f1_score_avg_method, upsampling):
    classifier = ct.Classifier(df, 'productivity_binned_binary', upsample=upsampling)
    #classifier.splitData()
    #classifier.upsampleTrainData()
    #classifier.downsampleTrainData()

    # get parameters for GridSearch
    # F1 Score with micro averaging
    score = classifier.f1(average=f1_score_avg_method) # macro = für jede klasse Durchschnitt der Berechnung der Scores
    # Classifier is Random Forest from sklearn
    estimator = classifier.randomForest()
    # use 10 fold cross validation as for other classifiers
    cv = classifier.fold(k=10, random_state=42)
 
    
    classifier.dropColumns([
            "original_title"
            ,"quarter"
            ,"productivity_binned_multi"
    ])
    
    classifier.dropColumnByPrefix("actor")
    classifier.dropColumnByPrefix("quarter")

    # contains all variables that GridSearch is comuting F1 score for
    param_grid = {"max_depth": [3, None],
                  "max_features": [1,3,10],
                  "min_samples_split": [2,3,10],
                  "min_samples_leaf": [1,3,10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    print(status + 'starting GridSearch')
    start = time()

    
    
    gs = classifier.gridSearch(estimator,
                               score,
                               param_grid,
                               print_results=False,
                               verbose=2,
                               cv=cv
                               #,onTrainSet=True
                               )
    classifier.gridSearchBestScore(gs)
    print("GridSearch for RandomForest took {}".format(time()-start))
    classifier.gridSearchResults2CSV(gs, param_grid, "randForest.csv")

    #save the best result - OPTIONAL
    #save_model(gs.best_params_, gs.best_score_, f1_score_avg_method, upsampling)
  
    print(status+"GridSearch for RandomForest took {} seconds".format("%.2f" % (float(time()-start))))
    #store to csv - OPTIONAL
    #classifier.gridSearchResults2CSV(gs, param_grid, "../../data/processed/randForest2.csv")
    print(status + 'CSV with scores saved successfully')
    
    
    # calculate cross validation: try samplings
    estimator.set_params(
        max_depth=None,
        max_features=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        criterion="entropy"
    )
    print(classifier.cross_validate(cv,estimator,sample=""))


    #run classifier on test set and print classification report
    #classifier.fit_predict(gs.best_estimator_)
    #print(classifier.confusion_matrix())
    #classifier.classification_report()
    
    
    
    
    return


def save_model(gs_params, gs_score, f1_score_avg_method, upsampling):
    filename = "ModelSelection_RandomForest_BestScore"
    #pickle.dump(modelRF, open(filename, "wb"))
    name = filename+".txt"
    #file = open(name,"w")
    try:
         with open(name, 'a') as file:
             #file.write("Best Score: "+str(modelRF))
             gs_score = "%.4f" % float(gs_score)
             file.write(str(datetime.datetime.now())+" | Score: "+str(gs_score)+" | Upsampling: "+str(upsampling)+" | F1 Average: "+f1_score_avg_method+" | "+str(gs_params)+"\n")   
             print(status + 'Best Score and Params saved to .txt file successfully')
    except ValueError:
         print("Oops!  File does not exist. Failed to write to csv")
    file.close()
    return


#import data set
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

#some strings we need
status = '[Status: ]'

#drop not needed features
df = df.drop(["original_title",
            "adult",
            "productivity_binned_multi"],
             #"production_companies"],
             1)
 
f1_score_avg_method='macro'
upsampling=True
RandomForest(df, f1_score_avg_method,upsampling)




"""
--------------------------- GRID SEARCH BEST SCORE ---------------------------
  Best score is 0.580698370927875 with params {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}.
------------------------------------------------------------------------------
"""