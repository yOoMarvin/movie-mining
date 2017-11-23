"""
Created on Thu Nov 16 13:07:06 2017

@author: Marvin
"""

# GOAL OF SCRIPT:
# place for all the best classifiers. Fit them here with best params
# Predict with cross validation
# Plot roc curve on all


from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder




# Import data
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)
print('data imported')

# Target
target = df['productivity_binned']

# Drop unwanted columns
columns_to_drop = [
        "original_title",
        "productivity_binned"
        ]
df = df.drop(columns_to_drop, axis=1)


# KStratifiedFold with random_state = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



##### KNN #####

print('#########   KNN EVALUATION   #########')

knn_estimator = KNeighborsClassifier(n_neighbors=3,
                            algorithm="auto",
                            weights="uniform",
                            p=2,
                            metric="manhattan")

# Fitting the model
knn_estimator.fit(df, target)
print('done doing the fit')

# Predict with cross validation
knn_scores = cross_val_score(knn_estimator, df, target, cv=cv, scoring='f1_micro')
print("KNN F1 Score: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))














###### ROC CURVES #####
#define function for computing average roc for cross validation
#see http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Function for computing the avg_roc curve for a specific label
def avg_roc(cv, estimator, data, target, pos_label):
    mean_fpr = np.linspace(0, 1, 100) # = [0.0, 0.01, 0.02, 0.03, ... , 0.99, 1.0]
    tprs = []
    aucs = []

    for train_indices, test_indices in cv.split(data, target):
        train_data = data[train_indices]
        train_target = target[train_indices]
        estimator.fit(train_data, train_target)

        test_data = data[test_indices]
        test_target = target[test_indices]
        decision_for_each_class = estimator.predict_proba(test_data)#have to use predict_proba or decision_function

        fpr, tpr, thresholds = roc_curve(test_target, decision_for_each_class[:,1], pos_label=pos_label)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0 # tprs[-1] access the last element
        aucs.append(auc(fpr, tpr))

        #plt.plot(fpr, tpr)# plot for each fold

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # set the last tpr to 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, mean_auc, std_auc


# Function for computing rov curve for all labels using micro measurements
def micro_roc(estimator, data, target):
    # Import some data to play with
    X = data
    y = target

    # Binarize the output
    y = label_binarize(y, classes=['no', 'yes']) #Adjust the labels to your need
    n_classes = y.shape[1]

    # shuffle and split training and test sets --> Need to to this, no cross val here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(estimator)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc











# PLOTTING THE CURVES FOR SPECIFIC LABELS
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal

# KNN / 3 NN - AVG for Label
mean_fpr, mean_tpr, mean_auc, std_auc = avg_roc(cv, knn_estimator, df.values, target.values, 'yes') #Take care of the label here! Is the binning label
plt.plot(mean_fpr, mean_tpr, label='3-NN (AUC: {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc))

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()



# PLOT CURVES FOR MICRO ROC
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal

# KNN / 3 NN - Micro Roc
fpr, tpr, roc_auc = micro_roc(knn_estimator, df, target)
plt.plot(fpr[2], tpr[2],lw=2, label='3-NN ROC curve (area = %0.2f)' % roc_auc[2])

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()
