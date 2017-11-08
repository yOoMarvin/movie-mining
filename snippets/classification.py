
# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(golf_encoded, golf['Play'])


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_estimator = KNeighborsClassifier(3)
#knn_estimator.fit....


# NEAREST CENTROID
from sklearn.neighbors.nearest_centroid import NearestCentroid
nearest_centroid_estimator = NearestCentroid()
#nearest_centroid_estimator.fit....


# EVALUATION
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

golf_prediction = ['yes','no','yes','yes','no','yes','yes','no','no','no','no','no','no','yes']

print(confusion_matrix(golf['Play'], golf_prediction))
print(accuracy_score(golf['Play'], golf_prediction))
print(classification_report(golf['Play'], golf_prediction))





# DECISION TREE
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree
decision_tree = tree.DecisionTreeClassifier(max_depth=2)#max_depth=2, because to see onl a small decision tree
decision_tree.fit(iris_binned_and_encoded, iris['Name'])

## VISUALIZE
import graphviz
from sklearn.utils.multiclass import unique_labels

dot_data = tree.export_graphviz(decision_tree,
                         feature_names=iris_binned_and_encoded.columns.values,
                         class_names=unique_labels(iris['Name']),
                         filled=True, rounded=True,special_characters=True,out_file=None)
graphviz.Source(dot_data)
