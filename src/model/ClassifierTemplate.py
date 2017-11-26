# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:07:06 2017

@author: Steff
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample



class Classifier:

    # data = pd.DataFrame containing the attributes
    # truth = pd.DataFrame containing the class
    # truth_arr = array of truth values

    def __init__(self,data,c,upsample=False):
        # set data
        self.data = data
        if upsample:
            self.upsample(c)
        # set truth array
        try:
            truth = np.array([x.decode('ascii') for x in self.data[c].values]) # "inline" for loop with []
        except AttributeError:
            truth = np.array(self.data[c].values)
            pass
        self.truth = pd.DataFrame(truth,columns=[c])
        self.dropColumns([c])
        self.extractTruthArray()

    """
        Methods for Preprocessing
    """

    # drop all given columns
    def dropColumns(self,columns):
        """
            Drop given columns from the classifier DataFrame
            :param columns: array of column names
        """
        self.data = self.data.drop(columns,axis=1)
    
    # drop all columns with given prefix
    def dropColumnByPrefix(self,prefix):
        #print(self.data.filter(regex=prefix))
        self.data.drop(list(self.data.filter(regex=prefix,axis=1)), axis=1, inplace=True)

    # drop all rows containing missing values
    def dropMissing(self):
        missing_indices = np.array([],dtype=int)
        for c in self.data.columns:
            zero = None
            if type(self.data[c][0]) == "<class 'bytes'>":
                zero = b'?'

            if zero is not None:
                try:
                    missing_indices = np.append(missing_indices,self.data[self.data[c]==zero].index.values)
                except TypeError:
                    print("TypeError",c,self.data[c][0],type(self.data[c][0]),isinstance(self.data[c][0],str),zero)
                    pass

        missing_indices = np.unique(missing_indices)
        self.data = self.data.drop(missing_indices)
        self.truth = self.truth.drop(missing_indices)
        self.extractTruthArray()

    #Upsample the minority class
    def upsample(self, c):
        # Separate majority and minority classes
        df_majority = self.data[self.data[c] == "yes"]
        df_minority = self.data[self.data[c] == "no"]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority),  # to match majority class
                                         random_state=42)  # reproducible results

        # Combine majority class with upsampled minority class
        self.data = pd.concat([df_majority, df_minority_upsampled])


    # HotEncode given columns
    def hotEncode(self,columns):
        data_encoded = pd.get_dummies(self.data[columns])
        self.dropColumns(columns)
        self.data = pd.concat([self.data, data_encoded],axis=1)

    # LabelEncode given columns
    def labelEncode(self,columns):
        data_encoded = self.data[columns].apply(preprocessing.LabelEncoder().fit_transform)
        self.dropColumns(columns)
        self.data = pd.concat([self.data, data_encoded],axis=1)

    # MinMaxScale given columns
    def scale(self,columns):
        scaler = preprocessing.MinMaxScaler()
        data_preprocessed = pd.DataFrame(
            scaler.fit_transform(self.data[columns]),
            columns=columns,
            index=self.data.index
        )
        self.dropColumns(columns)
        self.data = pd.concat([self.data, data_preprocessed],axis=1)

    def extractTruthArray(self):
        c, r = self.truth.values.shape
        self.truth_arr = self.truth.values.reshape(c,)

    # return # of rows in DataFrame
    def size(self):
        return len(self.data)

    # print distribution percentage of labels
    def balanceInfo(self,data = None):
        l = {}
        total = 0

        if (data is None):
            scalar = self.truth[self.truth.columns[0]]
        else:
            scalar = np.array(data)

        unique = set(scalar)

        for c in unique:
            l[c] = len(scalar[scalar==c])
            total += l[c]

        for c in l:
            print("class: {}, len = {} -> {}%".format( c,l[c],round(l[c]/total,2)*100 ))

    """
        Methods for Classification
    """

    def randomForest(self):
        return RandomForestClassifier()#n_estimators: nr of trees
    
    def knn(self):
        return KNeighborsClassifier()

    def bayes(self):
        return GaussianNB()

    def tree(self):
        return tree.DecisionTreeClassifier()

    def centroid(self):
        return NearestCentroid()

    def svc(self):
        return LinearSVC()
    
    def neuralnet(self):
        return MLPClassifier()

    def f1(self,pos_label = 1,average = 'binary'):
        # use first value of truth if not defined
        if (pos_label is None):
            pos_label = self.truth[self.truth.columns[0]].unique()[0]

        return make_scorer(
                f1_score,
                pos_label=pos_label,
                average=average
        )

    def fold(self,k=10,random_state=42,shuffle=True):
        return StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    def splitData(self, size = 0.2, random = 42):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data,
            self.truth_arr,
            test_size=size,
            random_state=random,
            stratify=self.truth_arr
        )

    def fit_predict(self,estimator):
        estimator.fit(self.data_train,self.target_train)
        self.predict = estimator.predict(self.data_test)

    def gridSearch(self,estimator,scoring,parameters = [],verbose=0,print_results=True,cv=None):
        print("starting GridSearch. Count of columns = {}".format( len(self.data.columns) ))
        print("columns: {}" .format( self.data.columns ))

        grid_search_estimator = GridSearchCV(estimator, parameters, scoring=scoring, verbose=verbose, cv=cv)
        grid_search_estimator.fit(self.data,self.truth_arr)

        if (print_results):
            print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))

        if (print_results):
            results = grid_search_estimator.cv_results_
            for i in range(len(results['params'])):
                print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))

        return grid_search_estimator

    def gridSearchBestScore(self,gs):
        print(
                "--------------------------- GRID SEARCH BEST SCORE ---------------------------\n",
                "Best score is {} with params {}.\n".format(gs.best_score_, gs.best_params_ ),
                "------------------------------------------------------------------------------\n"
        )

    def gridSearchResults2CSV(self,gs,parameters,filename):
        with open(filename,'w',newline="\n",encoding="utf-8") as csvfile:
            fieldnames = list(parameters.keys())
            fieldnames.append("result")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            results = gs.cv_results_
            for i in range(len(results['params'])):
                row = results['params'][i]
                row["result"] = results['mean_test_score'][i]
                writer.writerow(row)


    """
        Methods for Evaluation
        (needs evaluation ;))
    """

    def confusion_matrix(self):
        labels = list(set(self.target_test))
        return [confusion_matrix(self.target_test,self.predict,labels=labels),labels]

    def classification_report(self):
        print(classification_report(self.target_test,self.predict))

    def report(self,p):
        measures = {}
        row = 0
        for label_r in labels: #truth
            col = 0
            for label_c in labels: #prediction
                if (label_r == p) & (label_c == p):
                    measure = "tp"
                if (label_r == p) & (label_c != p):
                    measure = "fn"
                if (label_r != p) & (label_c == p):
                    measure = "fp"
                if (label_r != p) & (label_c != p):
                    measure = "tn"

                print("{} -> Truth: {}, Predict: {}, Amount: {}".format( measure,label_r,label_c,m[row][col] ))

                measures[measure] = m[row][col]
                col += 1
            row += 1

        accuracy = (measures["tp"]+measures["tn"]) / (measures["tp"]+measures["tn"]+measures["fp"]+measures["fn"])
        precision = (measures["tp"]) / (measures["tp"]+measures["fp"])
        recall = (measures["tp"]) / (measures["tp"]+measures["fn"])
        f1 = (2*precision*recall) / (precision + recall)
        accuracy = round(accuracy*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        print("Accuracy: {}%, Precision: {}%, Recall: {}%, F1: {}%".format( accuracy,precision,recall,f1 ))
