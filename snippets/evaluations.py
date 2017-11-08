# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
golf_data = golf_encoded
golf_target = golf['Play']
data_train, data_test, target_train, target_test = train_test_split(
    golf_data, golf_target,test_size=0.2, random_state=42, stratify=golf_target)
print("=======TRAIN=========")
print(data_train)
print(target_train)





# CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracy_iris = cross_val_score(decision_tree, iris_binned_and_encoded, iris['Name'], cv=10, scoring='accuracy')
accuracy_iris

## STRATIFIED
from sklearn.model_selection import StratifiedKFold
cross_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_each_split = cross_val_score(decision_tree, iris_binned_and_encoded, iris['Name'], cv=cross_val, scoring='accuracy')
acc_each_split.mean()

## PREDICTIONS FROM CROSS VALIDATION
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(decision_tree, iris_binned_and_encoded, iris['Name'], cv=10)
print(predicted)





# ROC CURVES
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

knn_estimator = KNeighborsClassifier(3)

data_train, data_test, target_train, target_test = train_test_split(credit_data, credit_target)
knn_estimator.fit(data_train, target_train)
proba_for_each_class = knn_estimator.predict_proba(data_test)#have to use predict_proba or decision_function

fpr, tpr, thresholds = roc_curve(target_test, proba_for_each_class[:,1], pos_label='good')#choose the second class

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8) # draw diagonal
plt.plot(fpr,tpr,label='K-NN')

plt.legend()
plt.show()


## AVG ROC CURVES
from scipy import interp
from sklearn.metrics import roc_curve, auc

def avg_roc(cv, estimator, data, target, pos_label):
    mean_fpr = np.linspace(0, 1, 100) # = [0.0, 0.01, 0.02, 0.03, ... , 0.99, 1.0]
    tprs = []
    aucs = []
    for train_indices, test_indices in cv.split(data, target):
        train_data, train_target = data[train_indices], target[train_indices]
        estimator.fit(train_data, train_target)

        test_data, test_target = data[test_indices], target[test_indices]
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






# ADDING NOISE
import random
from sklearn.utils.multiclass import unique_labels
def add_noise(raw_target, percentage):
    labels = unique_labels(raw_target)
    target_with_noise = []
    for one_target_label in raw_target:
        if random.randint(1,100) <= percentage:
            target_with_noise.append(next(l for l in labels if l != one_target_label))
        else:
            target_with_noise.append(one_target_label)
    return target_with_noise
