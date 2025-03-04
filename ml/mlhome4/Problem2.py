import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#%%
# read data
wordlist = np.loadtxt('wordlist.txt', dtype=str)
documents = np.loadtxt('documents.txt', dtype=int)
newsgroups = np.loadtxt('newsgroups.txt', dtype=str)
groupnames = np.loadtxt('groupnames.txt', dtype=str)

# Initialize the occurrence matrix
num_postings = 16242
num_keywords = 100
X = np.zeros((num_postings, num_keywords))

# Fill the occurrence matrix
for doc_id, keyword_id, value in documents:
    X[doc_id - 1, keyword_id - 1] = value

# Map newsgroups to integers
unique_groups = np.unique(newsgroups)
group_map = {group: i for i, group in enumerate(unique_groups)}
y = np.array([group_map[group] for group in newsgroups])
#%%
# Q1 
# Build a random forest, report the 5-fold cv error. 
# How many predictors are chosen in each tree. How many trees are used.
# Report the best CV error, confusion matrix and tuning parameters. 
# The ten most important keywords based on variable importance
print("----------Random Forest---------")
param_grid = {'n_estimators': [50,100,150,200]}
rf = RandomForestClassifier(max_features='sqrt', random_state=14,n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
print('Tuning parameters:\n')
print(best_rf.get_params())
# Cross-validation
rf_cv_scores = cross_val_score(best_rf, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
rf_best_cv_error = 1 - np.mean(rf_cv_scores)
rf_conf_matrix = confusion_matrix(y, best_rf.predict(X))
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=rf_conf_matrix, display_labels=groupnames)
disp.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.savefig('P2Q1.jpg',dpi=400)

print("Best CV Error:", rf_best_cv_error)
print("Top 10 Important Keywords:", [wordlist[i] for i in best_rf.feature_importances_.argsort()[-10:][::-1]])

#%%
#Q2
# Build a boosting tree. Report the 5-fold cv error. 
# report the best CV error, confusion matrix and tuning parameters. 
'''
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=14)
gbm_cv_scores = cross_val_score(gbm, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
'''
print("---------boosting tree----------")
gbm = GradientBoostingClassifier(max_features='sqrt', random_state=14)
grid_search = GridSearchCV(gbm, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
print('Tuning parameters:\n')
print(best_rf.get_params())
gbm_cv_scores = cross_val_score(best_rf, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
gbm_best_cv_error = 1 - np.mean(gbm_cv_scores)
gbm_conf_matrix = confusion_matrix(y, gbm.fit(X, y).predict(X))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=gbm_conf_matrix, display_labels=groupnames)
disp.plot(cmap=plt.cm.Blues)
plt.title("Boosting Tree Confusion Matrix")
plt.savefig('P2Q2.jpg',dpi=400)
print("Best CV Error:", gbm_best_cv_error)

#%%
# Q4
# Linear Discriminant Analysis
print("---------LDA----------")
lda = LinearDiscriminantAnalysis()
lda_cv_scores = cross_val_score(lda, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
lda_best_cv_error = 1 - np.mean(lda_cv_scores)
lda_conf_matrix = confusion_matrix(y, lda.fit(X, y).predict(X))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=lda_conf_matrix, display_labels=groupnames)
disp.plot(cmap=plt.cm.Blues)
plt.title("LDA Confusion Matrix")
plt.savefig('P2Q4.jpg',dpi=400)
print("Best CV Error:", lda_best_cv_error)

#%%
# Q5
# Quadratic Discriminant Analysis
print("---------QDA----------")
qda = QuadraticDiscriminantAnalysis()
qda_cv_scores = cross_val_score(qda, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
qda_best_cv_error = 1 - np.mean(qda_cv_scores)
qda_conf_matrix = confusion_matrix(y, qda.fit(X, y).predict(X))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=qda_conf_matrix, display_labels=groupnames)
disp.plot(cmap=plt.cm.Blues)
plt.title("QDA Confusion Matrix")
plt.savefig('P2Q5.jpg',dpi=400)
print("Best CV Error:", qda_best_cv_error)

#%%
# Q6
print('\n-------------------------------------')
print("Random Forest CV Error:", rf_best_cv_error)
print("Boosting Tree CV Error:", gbm_best_cv_error)
print("LDA CV Error:", lda_best_cv_error)
print("QDA CV Error:", qda_best_cv_error)