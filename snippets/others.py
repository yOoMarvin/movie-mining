
# GRID SEARCH
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

knn_estimator = KNeighborsClassifier()
parameters = {
    'n_neighbors':[2,3,4,5,6,7,8],
    'algorithm':['ball_tree', 'kd_tree', 'brute']
}
stratified_10_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search_estimator = GridSearchCV(knn_estimator, parameters, scoring='accuracy', cv=stratified_10_fold_cv)
grid_search_estimator.fit(iris_data,iris_target)# this will try out all possibilities

#one can use the best estimator for further prediction
#this estimator is trained on the whole dataset with the best hyper parameters
#grid_search_estimator.best_estimator_.predict()
print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))

results = grid_search_estimator.cv_results_

#import pprint
#pprint.pprint(results)

for i in range(len(results['params'])):
    print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))
