import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#%%
# Load data
train_data = pd.read_csv('train_resized.csv')
test_data = pd.read_csv('test_resized.csv')

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)

#%%
# Q1
# Use 3 and 6 from train and test to build an SVM classifier
# Use a linear kernel and choose the best cost parameter by 5 fold CV. 
# report the misclassification error, confusion matrix on test.
# report the time cost of training your model
print("-------------Linear SVM for Digits 3 and 6------------")
train_filter = (y_train == 3) | (y_train == 6)
test_filter = (y_test == 3) | (y_test == 6)

X_train_36, y_train_36 = X_train[train_filter], y_train[train_filter]
X_test_36, y_test_36 = X_test[test_filter], y_test[test_filter]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_36 = imputer.fit_transform(X_train_36)
X_test_36 = imputer.transform(X_test_36)

# Linear SVM
param_grid_linear = {'C': range(1,100,10)}
linear_svc = SVC(kernel='linear')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=14)

# Grid search with cross-validation
grid_linear = GridSearchCV(linear_svc, param_grid_linear, cv=cv, scoring='accuracy')
start_time = time.time()
grid_linear.fit(X_train_36, y_train_36)
end_time = time.time()

# Evaluate
best_linear_model = grid_linear.best_estimator_
y_pred_36 = best_linear_model.predict(X_test_36)
misclassification_error_36 =  1 - accuracy_score(y_test_36, y_pred_36)
conf_matrix_36 = confusion_matrix(y_test_36, y_pred_36)

# Display confusion matrix
disp_36 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_36, display_labels=[3, 6])
disp_36.plot(cmap='Blues')
plt.title("Linear SVM Confusion Matrix for Digits 3 and 6")
plt.show()


print("Best Cost Parameter (C):", grid_linear.best_params_['C'])
print("Misclassification Error:", misclassification_error_36)
print("Confusion Matrix:\n", conf_matrix_36)
print("Time Cost of Training (seconds):", end_time - start_time)
#%%
# Q2

print("---------Radial SVM for Digits 3 and 6-----------")
param_grid_rbf = {'C': [0.1,1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
rbf_svc = SVC(kernel='rbf')

# Impute missing values for radial SVM
X_train_36 = imputer.fit_transform(X_train_36) 
X_test_36 = imputer.transform(X_test_36)

# Grid search with cross-validation
grid_rbf = GridSearchCV(rbf_svc, param_grid_rbf, cv=cv, scoring='accuracy')
start_time = time.time()
grid_rbf.fit(X_train_36, y_train_36)
end_time = time.time()

# Evaluate
best_rbf_model = grid_rbf.best_estimator_
y_pred_36_rbf = best_rbf_model.predict(X_test_36)
misclassification_error_36_rbf = 1 - accuracy_score(y_test_36, y_pred_36_rbf)
conf_matrix_36_rbf = confusion_matrix(y_test_36, y_pred_36_rbf)

# Display confusion matrix
disp_36_rbf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_36_rbf, display_labels=[3, 6])
disp_36_rbf.plot(cmap='Blues')
plt.title("Radial SVM Confusion Matrix for Digits 3 and 6")
plt.show()


print("Best Cost Parameter (C):", grid_rbf.best_params_['C'])
print("Best Cost Parameter (gamma):", grid_rbf.best_params_['gamma'])
print("Misclassification Error:", misclassification_error_36_rbf)
print("Confusion Matrix:\n", conf_matrix_36_rbf)
print("Time Cost of Training (seconds):", end_time - start_time)

#%%
# Q3
print("--------Comparison for Digits 3 and 6--------")
print("Linear SVM Error:", misclassification_error_36)
print("Radial SVM Error:", misclassification_error_36_rbf)
#%%
# Q4
# Filter data for digits 1, 2, 5, and 8
print("-------Linear SVM for Digits 1, 2, 5, and 8-------")
train_filter_1258 = (y_train == 1) | (y_train == 2) | (y_train == 5) | (y_train == 8)
test_filter_1258 = (y_test == 1) | (y_test == 2) | (y_test == 5) | (y_test == 8)

X_train_1258, y_train_1258 = X_train[train_filter_1258], y_train[train_filter_1258]
X_test_1258, y_test_1258 = X_test[test_filter_1258], y_test[test_filter_1258]

# Impute missing values for digits 1, 2, 5, and 8
X_train_1258 = imputer.fit_transform(X_train_1258)
X_test_1258 = imputer.transform(X_test_1258)

# Linear SVM for multi-class
grid_linear_1258 = GridSearchCV(linear_svc, param_grid_linear, cv=cv, scoring='accuracy')
start_time = time.time()
grid_linear_1258.fit(X_train_1258, y_train_1258)
end_time = time.time()

# Evaluate
best_linear_model_1258 = grid_linear_1258.best_estimator_
y_pred_1258 = best_linear_model_1258.predict(X_test_1258)
misclassification_error_1258 = np.mean(y_pred_1258 != y_test_1258)
conf_matrix_1258 = confusion_matrix(y_test_1258, y_pred_1258)

# Display confusion matrix
disp_1258 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_1258, display_labels=[1, 2, 5, 8])
disp_1258.plot(cmap='Blues')
plt.title("Linear SVM Confusion Matrix for Digits 1, 2, 5, and 8")
plt.show()

print("Best CV Parameters:", grid_linear_1258.best_params_)
print("Misclassification Error:", misclassification_error_1258)
print("Confusion Matrix:\n", conf_matrix_1258)
print("Time Cost of Training (seconds):", end_time - start_time)


#%%
# Q5
print("-------Multi-class SVM for All Digits----------")
param_grid_full = {'C': [1, 10], 'gamma': [0.01, 0.1]}
X_train = imputer.fit_transform(X_train) 
X_test = imputer.transform(X_test)

grid_full = GridSearchCV(rbf_svc, param_grid_full, cv=cv, scoring='accuracy')
start_time = time.time()
grid_full.fit(X_train, y_train)
end_time = time.time()

# Evaluate
best_full_model = grid_full.best_estimator_
y_pred_full = best_full_model.predict(X_test)
misclassification_error_full = np.mean(y_pred_full != y_test)
conf_matrix_full = confusion_matrix(y_test, y_pred_full)

# Display confusion matrix
disp_full = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_full, display_labels=np.unique(y_test))
disp_full.plot(cmap='Blues')
plt.title("Multi-class SVM Confusion Matrix for All Digits")
plt.show()

print("Best CV Parameters:", grid_full.best_params_)
print("Misclassification Error:", misclassification_error_full)
print("Confusion Matrix:\n", conf_matrix_full)
print("Time Cost of Training (seconds):", end_time - start_time)