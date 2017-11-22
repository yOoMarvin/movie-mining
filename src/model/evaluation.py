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
        "productivity_binned",
        "id"
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

def avg_roc(cv, estimator, data, target, pos_label):
    mean_fpr = np.linspace(0, 1, 100) # = [0.0, 0.01, 0.02, 0.03, ... , 0.99, 1.0]
    tprs = []
    aucs = []

    for train_indices, test_indices in cv.split(data, target):
        train_data = data[train_indices]
        train_target = target[train_indices]
        estimator.fit(train_data, train_target)

        test_data = data[train_indices]
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


# PLOTTING THE CURVE
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal

# KNN / 3 NN
mean_fpr, mean_tpr, mean_auc, std_auc = avg_roc(cv, knn_estimator, df, target, 'good')
plt.plot(mean_fpr, mean_tpr, label='3-NN (AUC: {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc))

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()
