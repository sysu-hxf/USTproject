import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# read data 
train_data = pd.read_csv('BreastCancer_train.csv')
test_data  = pd.read_csv('BreastCancer_test.csv')

# drop nan
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

"""
X_train: feature data in training set, we do not need ID(useless) and Class(Label, not feature)
y_train: label data in training set. We map "benign" and "malignant" into 0 and 1
X_test : feature data in test set
y_test : label data in test set
"""
X_train = train_data.drop(columns=['Id', 'Class'])
y_train = train_data['Class'].map({'benign': 0, 'malignant': 1})
X_test = test_data.drop(columns=['Id','Class'])
y_test = test_data['Class'].map({'benign': 0, 'malignant': 1})

#%%
# Q1
# Use all the predictors to fit a logistic regression model
# Report the summary and plot the ROC curve on the test dataset.

model_all = LogisticRegression()
model_all.fit(X_train, y_train)
y_score   = model_all.predict_proba(X_test)[:, 1]

# print model summary
y_pred = model_all.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)
print("----The Logistic model with all the predictors:----\n")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

# cacaulate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc              = auc(fpr, tpr)

# draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('P2Q1.jpg',dpi=400)

#%%
#Q2
# Use the predictors Cl.thickness, Cell.shape, Marg.adhesion, Bare.nuclei, Bl.cromatin to fit a logistic model
# Report the summary and plot the ROC curve on the test dataset.
# select features
X_train_subset = train_data[['Cl.thickness', 'Cell.shape', 'Marg.adhesion', 'Bare.nuclei', 'Bl.cromatin']]
X_test_subset = test_data[['Cl.thickness', 'Cell.shape', 'Marg.adhesion', 'Bare.nuclei', 'Bl.cromatin']]

model_subset = LogisticRegression()
model_subset.fit(X_train_subset, y_train)

y_score_subset = model_subset.predict_proba(X_test_subset)[:, 1]
# print model summary
y_pred = model_subset.predict(X_test_subset)
cm     = confusion_matrix(y_test, y_pred)
print("----The Logistic model with the selected predictors:----\n")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

fpr_subset, tpr_subset, thresholds_subset = roc_curve(y_test, y_score_subset)
roc_auc_subset = auc(fpr_subset, tpr_subset)

plt.figure()
plt.plot(fpr_subset, tpr_subset, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_subset)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Subset')
plt.legend(loc="lower right")
plt.savefig('P2Q2.jpg',dpi=400)

#%%
# Q3
# Use all the predictors to fit an LDA model and report the summary. Plot the ROC curve on the test dataset.
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

y_score_lda = lda_model.predict_proba(X_test)[:, 1]

y_pred = lda_model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)
print("----The LDA model with all the predictors:----\n")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))


fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_test, y_score_lda)
roc_auc_lda = auc(fpr_lda, tpr_lda)

plt.figure()
plt.plot(fpr_lda, tpr_lda, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_lda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - LDA')
plt.legend(loc="lower right")
plt.savefig('P2Q3.jpg',dpi=400)
#%%
# Q4
# Use the predictors Cl.thickness, Cell.shape, Marg.adhesion, Bare.nuclei, Bl.cromatin to fit an LDA model
# Report the summary and plot the ROC curve on the test dataset.

lda_model_subset = LinearDiscriminantAnalysis()
lda_model_subset.fit(X_train_subset, y_train)

y_score_lda_subset = lda_model_subset.predict_proba(X_test_subset)[:, 1]

y_pred = lda_model_subset.predict(X_test_subset)
cm     = confusion_matrix(y_test, y_pred)
print("----The LDA model with the selected predictors:----\n")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

fpr_lda_subset, tpr_lda_subset, thresholds_lda_subset = roc_curve(y_test, y_score_lda_subset)
roc_auc_lda_subset = auc(fpr_lda_subset, tpr_lda_subset)

plt.figure()
plt.plot(fpr_lda_subset, tpr_lda_subset, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_lda_subset)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - LDA Subset')
plt.legend(loc="lower right")
plt.savefig('P2Q4.jpg',dpi=400)

#%%
# Q5
# Use all the predictors to fit a QDA model and report the summary. Plot the ROC curve on the test dataset
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)

y_score_qda = qda_model.predict_proba(X_test)[:, 1]

y_pred = qda_model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)
print("----The QDA model with all the predictors:----\n")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

fpr_qda, tpr_qda, thresholds_qda = roc_curve(y_test, y_score_qda)
roc_auc_qda = auc(fpr_qda, tpr_qda)

plt.figure()
plt.plot(fpr_qda, tpr_qda, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_qda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - QDA')
plt.legend(loc="lower right")
plt.savefig('P2Q5.jpg',dpi=400)

#%%
# Compare by AUC
print(f"AUC for Logistic Regression (All Features): \n{roc_auc}")
print(f"AUC for Logistic Regression (Subset): \n{roc_auc_subset}")
print(f"AUC for LDA (All Features): \n{roc_auc_lda}")
print(f"AUC for LDA (Subset): \n{roc_auc_lda_subset}")
print(f"AUC for QDA (ALL Features): \n{roc_auc_qda} ")