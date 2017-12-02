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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Custom imports of classifications
import ClassifierTemplate as ct


### TODO
### grab grid search parameter results of each classifier (from comment in model selection)
### Grab the data which was used (dropped columns etc.)
### Fit the model here with the cv and use it for roc curves

# Import data once
data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)
# DataFrame containing label (!)
df = pd.DataFrame(data)


# KStratifiedFold with random_state = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)






##### NAIVE BAYES #####
print('#########   NAIVE BAYES EVALUATION   #########')

# Build Classifier object with DataFrame and column name of truth values
nb = ct.Classifier(df,"productivity_binned_binary", False)
nb.dropColumns([
        "original_title"
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])


### drop columns by prefix if needed
nb.dropColumnByPrefix("actor")
nb.dropColumnByPrefix("director")
nb.dropColumnByPrefix("country")
nb.dropColumnByPrefix("genre")

# Put values into clear variable names
naivebayes_data = nb.data
naivebayes_target = nb.truth_arr
naivebayes_estimator = GaussianNB()

naivebayes_estimator.fit(naivebayes_data, naivebayes_target)

naivebayes_scores = cross_val_score(naivebayes_estimator,
                                    naivebayes_data,
                                    naivebayes_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("Naive Bayes F1 Score: %0.2f (+/- %0.2f)" % (naivebayes_scores.mean(), naivebayes_scores.std() * 2))


##### KNN #####
print('#########   K-NN EVALUATION   #########')
# Build Classifier object with DataFrame and column name of truth values
knn = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
knn.dropColumns([
        "original_title"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
        ,"quarter"

        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
])

### drop columns by prefix if needed
knn.dropColumnByPrefix("actor")
knn.dropColumnByPrefix("country")
knn.dropColumnByPrefix("genre")
knn.dropColumnByPrefix("quarter_")
### thresholds
knn.thresholdByColumn(3,"company")
knn.thresholdByColumn(8,"actor")
knn.thresholdByColumn(3,"director")

knn_data = knn.data
knn_target = knn.truth_arr
knn_estimator = KNeighborsClassifier(algorithm='auto',
                                    metric='euclidean',
                                    n_neighbors=16,
                                    p=2,
                                    weights='uniform')

knn_estimator.fit(knn_data, knn_target)

knn_scores = cross_val_score(knn_estimator,
                                    knn_data,
                                    knn_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("16-NN F1 Score: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))


##### DECISION TREE #####
print('#########   DECISION TREE EVALUATION   #########')
# Build Classifier object with DataFrame and column name of truth values
tr = ct.Classifier(df,"productivity_binned_binary", False)

# drop columns not needed for Classification
tr.dropColumns([
        "original_title"
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

### scale something if needed
tr.scale([
        "budget"
])

tr.thresholdByColumn(3,"company")
tr.thresholdByColumn(8,"actor")
tr.thresholdByColumn(3,"director")

tree_data = tr.data
tree_target = tr.truth_arr
tree_estimator = tree.DecisionTreeClassifier(class_weight=None,
                                        criterion='entropy',
                                        max_depth=100,
                                        min_samples_split=5)

tree_estimator.fit(tree_data, tree_target)

tree_scores = cross_val_score(tree_estimator,
                                    tree_data,
                                    tree_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("Decision Tree F1 Score: %0.2f (+/- %0.2f)" % (tree_scores.mean(), tree_scores.std() * 2))


##### NEURAL NET #####
print('#########   NEURAL NET EVALUATION   #########')
# Build Classifier object with DataFrame and column name of truth values
net = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
net.dropColumns([
        "original_title"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"

        #,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        #,"year"
        #,"quarter"
])

### Thresholds
net.thresholdByColumn(3,"company")
net.thresholdByColumn(8,"actor")
net.thresholdByColumn(3,"director")

net_data = net.data
net_target = net.truth_arr
net_estimator = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=100,
                            activation='logistic',
                            alpha=0.0001,
                            max_iter=200)

net_estimator.fit(net_data, net_target)

net_scores = cross_val_score(net_estimator,
                                    net_data,
                                    net_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("Neural Net F1 Score: %0.2f (+/- %0.2f)" % (net_scores.mean(), net_scores.std() * 2))




##### RANDOM FOREST #####
print('#########   RANDOM FOREST EVALUATION   #########')
# Build Classifier object with DataFrame and column name of truth values
rf = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
rf.dropColumns([
        "original_title"
        ,"quarter"
        ,"productivity_binned_multi"
        , "adult"
])

rf.dropColumnByPrefix("belongs")
rf.dropColumnByPrefix("actor")
rf.dropColumnByPrefix("director")
rf.dropColumnByPrefix("production_country")

rf.thresholdByColumn(3,"company")
rf.thresholdByColumn(8,"actor")
rf.thresholdByColumn(3,"director")


rf_data = rf.data
rf_target = rf.truth_arr
rf_estimator = RandomForestClassifier(max_depth=None,
                                        max_features=5,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        bootstrap=True,
                                        criterion="entropy")

rf_estimator.fit(rf_data, rf_target)

rf_scores = cross_val_score(rf_estimator,
                                    rf_data,
                                    rf_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("Random Forest F1 Score: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))




##### SUPPORT VECTOR MACHINES #####
print('#########   SUPPORT VECTOR MACHINES EVALUATION   #########')
# Build Classifier object with DataFrame and column name of truth values
sv = ct.Classifier(df,"productivity_binned_binary")

sv.dropColumns([
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
sv.scale([
        "budget"
])

### drop columns by prefix if needed
sv.dropColumnByPrefix("actor_")
sv.dropColumnByPrefix("country")
sv.dropColumnByPrefix("genre")
sv.dropColumnByPrefix("quarter_")

sv.thresholdByColumn(3,"company")
sv.thresholdByColumn(8,"actor")
sv.thresholdByColumn(3,"director")

sv_data = sv.data
sv_target = sv.truth_arr
sv_estimator = LinearSVC(multi_class='ovr',
                        class_weight='balanced')

sv_estimator.fit(sv_data, sv_target)

sv_scores = cross_val_score(sv_estimator,
                                    sv_data,
                                    sv_target,
                                    cv=cv,
                                    scoring='f1_macro')

print("SVM F1 Score: %0.2f (+/- %0.2f)" % (sv_scores.mean(), sv_scores.std() * 2))














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
def macro_roc(estimator, data, target):
    # Data
    X = data
    y = target

    # Binarize the output
    y = label_binarize(y, classes=['yes', 'no']) #Adjust the labels to your need
    n_classes = y.shape[1]

    # shuffle and split training and test sets --> Need to to this, no cross val here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(estimator)
    if estimator == sv_estimator:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #for i in range(n_classes):
    #    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #    roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    if estimator == sv_estimator:
        fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    else:
        fpr["macro"], tpr["macro"], _ = roc_curve(y_test, y_score[:,1])
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc











# PLOTTING THE CURVES FOR SPECIFIC LABELS
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal

# Naive Bayes - AVG for Label
mean_fpr, mean_tpr, mean_auc, std_auc = avg_roc(cv, naivebayes_estimator, naivebayes_data.values, naivebayes_target, 'yes') #Take care of the label here! Is the binning label
plt.plot(mean_fpr, mean_tpr, label='Naive Bayes (AUC: {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc))

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()



# PLOT CURVES FOR MICRO ROC
plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal

# Naive Bayes - Macro Roc
fpr, tpr, roc_auc = macro_roc(naivebayes_estimator, naivebayes_data, naivebayes_target)
plt.plot(fpr['macro'], tpr['macro'],lw=2, label='Naive Bayes(ROC Area = %0.2f)' % roc_auc['macro'])
# KNN - Macro Roc
fpr, tpr, roc_auc = macro_roc(knn_estimator, knn_data, knn_target)
plt.plot(fpr['macro'], tpr['macro'],lw=2, label='16-NN(ROC Area = %0.2f)' % roc_auc['macro'])
# Decision Tree - Macro Roc
fpr, tpr, roc_auc = macro_roc(tree_estimator, tree_data, tree_target)
plt.plot(fpr['macro'], tpr['macro'],lw=2, label='Decision Tree(ROC Area = %0.2f)' % roc_auc['macro'])
# Neural Net - Macro Roc
#fpr, tpr, roc_auc = macro_roc(net_estimator, net_data, net_target)
#plt.plot(fpr['macro'], tpr['macro'],lw=2, label='Neural Net(ROC Area = %0.2f)' % roc_auc['macro'])
# Random Forest - Macro Roc
fpr, tpr, roc_auc = macro_roc(rf_estimator, rf_data, rf_target)
plt.plot(fpr['macro'], tpr['macro'],lw=2, label='Random Forest(ROC Area = %0.2f)' % roc_auc['macro'])
# Support Vector Machine - Macro Roc
fpr, tpr, roc_auc = macro_roc(sv_estimator, sv_data, sv_target)
plt.plot(fpr['macro'], tpr['macro'],lw=2, label='SVM(ROC Area = %0.2f)' % roc_auc['macro'])

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()
