import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv('cars.csv')
speed = data['speed'].values
dist = data['dist'].values
#%%
# Q1
loo = LeaveOneOut()
poly_degrees = range(1, 11)  
leaveone_errors = []

for degree in poly_degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(speed.reshape(-1, 1))
    model = np.linalg.lstsq(X_poly, dist,rcond=None)[0]
    errors = []
    for train_index, test_index in loo.split(speed):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = dist[train_index], dist[test_index]
        model = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        pred = X_test.dot(model)
        errors.append(mean_squared_error(y_test, pred))
    leaveone_errors.append(np.mean(errors))
fig = plt.figure(figsize=[6,6])
plt.plot(poly_degrees, leaveone_errors, marker='o')
plt.xlabel('Degree of Polynomial')
plt.ylabel('LOO Error')
plt.title('LOO Error vs. Degree of Polynomial')
plt.grid()
plt.savefig('P3Q1.jpg',dpi=400)
#%%
# Q2
kf = KFold(n_splits=5, shuffle=True, random_state=14)

cv_errors_5fold = []

for degree in poly_degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(speed.reshape(-1, 1))
    errors = []
    for train_index, test_index in kf.split(speed):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = dist[train_index], dist[test_index]
        model = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        pred = X_test.dot(model)
        errors.append(mean_squared_error(y_test, pred))
    cv_errors_5fold.append(np.mean(errors))
fig = plt.figure(figsize=[6,6])
plt.plot(poly_degrees, cv_errors_5fold, marker='o')
plt.xlabel('Degree of Polynomial')
plt.ylabel('5-Fold CV Error')
plt.title('5-Fold CV Error vs. Degree of Polynomial')
plt.grid()
plt.savefig('P3Q2.jpg',dpi=400)
#%%
# Q3
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
bandwidths = np.linspace(0.1, 10, 30)
loo_errors = []
cv_errors  = []

for para in bandwidths:
    model = KNeighborsRegressor(weights=lambda x: np.exp(-x**2 / (2 * para**2)))
    loo_folderror = []
    cv_folderror  = []
    
    mse_scorer    = make_scorer(mean_squared_error)
    loo_folderror = cross_val_score(model, speed.reshape(-1,1), dist, cv=loo, scoring=mse_scorer)
    loo_errors.append(np.mean(loo_folderror))
    cv_folderror  = cross_val_score(model, speed.reshape(-1,1), dist, cv=5, scoring=mse_scorer)
    cv_errors.append(np.mean(cv_folderror))

fig = plt.figure(figsize=[12,6])
plt.subplot(121)
plt.plot(bandwidths, loo_errors, marker='o', label = 'leave-one-out Error')
plt.legend()
plt.xlabel('Bandwidth')
plt.ylabel('leave-one-out Error')
plt.grid()
plt.subplot(122)
plt.plot(bandwidths, cv_errors,  marker='o', label = 'CV Error')
plt.legend()
plt.xlabel('Bandwidth')
plt.ylabel('5-Fold Cross validation Error')
plt.grid()
plt.savefig('P3Q3.jpg',dpi=400)

