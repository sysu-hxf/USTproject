import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
# loading data
data = pd.read_csv('./trees.csv')
print(data.isna().sum())
X = data['Girth'].values.reshape(-1, 1)
y = data['Volume'].values
# %%
# Q1
#  Fit polynomial models (deg=1,2,3,4) to predict the Volume using Girth.
#  Choose the model with the largest adjust R-squared. Plot with 2 standard error.
#  How about using 5-CV error.
print("--------Q1: Adjusted R^2---------\n")
models = []
degrees = [1, 2, 3, 4]
best_adj_r2 = 0
best_model = None
best_degree = 0

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    adj_r2 = 1 - (1 - r2_score(y, y_pred)) * (len(y) - 1) / (len(y) - degree - 1)
    models.append((model, poly_features, adj_r2))
    if adj_r2 > best_adj_r2:
        best_adj_r2 = adj_r2
        best_model = model
        best_degree = degree
        best_X_poly = X_poly

print(f"Best model degree: {best_degree}, Adjusted R-squared: {best_adj_r2}\n")

y_pred = best_model.predict(best_X_poly)
y_std = np.std(best_model.predict(best_X_poly) - y)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='data points')
plt.plot(X, y_pred, color='crimson', label='Best Fit Line')
plt.fill_between(X.ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std, color='pink', alpha=0.3, label='±2 Standard Error')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.title('Polynomial Regression')
plt.legend()
plt.grid('on')
plt.savefig('Problem2Q1(adj-R).jpg', dpi=400)

print("------------Q1: 5-fold CV--------------\n")
# define 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=1)
cv_error = []
for degree in degrees:
    mse = []
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    for train_index, test_index in kf.split(X_poly):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse.append(mean_squared_error(y_test, predictions))

    cv_error.append(np.mean(mse))
    print(f"Degree {degree}, CV error with 5-fold: {np.mean(mse)}\n")

best_degree = degrees[np.argmin(cv_error)]
print("Best model degree with 5-fold CV is:", best_degree)

best_poly_features = PolynomialFeatures(degree=best_degree)
X_poly = best_poly_features.fit_transform(X)
best_model = LinearRegression()
best_model.fit(X_poly, y)

y_pred = best_model.predict(X_poly)
y_std = np.std(best_model.predict(X_poly) - y)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='data points')
plt.plot(X, y_pred, color='crimson', label='best-fit line')
plt.fill_between(X.ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std, color='pink', alpha=0.3, label='±2 Standard Error')
plt.title('Best Polynomial Regression (5-fold CV)')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.legend()
plt.grid('on')
plt.savefig('Problem2Q1(5fold).jpg', dpi=400)
# %%
# Q2
# polynomial logistic regression model with deg=2
# predict whether the Volume is larger or not than 30
# Plot the function P(Volume > 30) with respect to Girth with 2 standard error.
print("------------Q2---------------\n")
from sklearn.linear_model import LogisticRegression

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
model = LogisticRegression()
model.fit(X_poly, (y > 30).astype(int))

# 绘制函数P(Volume > 30)与Girth的关系
girth_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
girth_range_poly = poly_features.transform(girth_range)
y_prob = model.predict_proba(girth_range_poly)[:, 1]

se_prob = np.sqrt(y_prob * (1 - y_prob) / len(y))
band_upper = y_prob + 2 * se_prob
band_lower = y_prob - 2 * se_prob

plt.figure(figsize=(10, 6))
plt.plot(girth_range, y_prob, label='P(Volume > 30)')
plt.fill_between(girth_range.ravel(), band_lower, band_upper, color='pink', alpha=0.3, label='±2 Standard Error')
plt.xlabel('Girth')
plt.ylabel('P(Volume > 30)')
plt.title('Polynomial Logistic Regression')
plt.legend()
plt.grid('on')
plt.savefig('Problem2Q2.jpg', dpi=400)

# %%
# Q3
# regression spline with deg=2
# to predict the Volume using the Girth at knots 10, 14, 18.
# Plot with 2 standard error

print("------------Q3---------------\n")
from patsy import dmatrix
from statsmodels.api import OLS

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

knots = [10, 14, 18]
transformed_x = dmatrix("bs(X, knots=knots, degree=2, include_intercept=False)", {"X": X}, return_type='dataframe')

spline_model = OLS(y, transformed_x).fit()

transformed_x_range = dmatrix("bs(X_range, knots=knots, degree=2, include_intercept=False)", {"X_range": X_range},
                              return_type='dataframe')
y_spline_pred = spline_model.predict(transformed_x_range)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='data points')
plt.plot(X_range, y_spline_pred, color='crimson', label='regression spline')

se_spline = np.sqrt(np.sum((y - spline_model.predict(transformed_x)) ** 2) / (len(y) - len(knots) - 1))
confidence_band_upper = y_spline_pred + 2 * se_spline
confidence_band_lower = y_spline_pred - 2 * se_spline

plt.fill_between(X_range.flatten(), confidence_band_lower, confidence_band_upper, color='pink', alpha=0.2,
                 label='±2 SE')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.legend()
plt.title('Regression Spline with 2-degree')
plt.grid('on')
plt.savefig('Problem2Q3.jpg', dpi=400)

# %%
# Q4
# smoothing spline to predict the Volume using the variable Girth
# smoothing level is chosen by Cross-Validation.
# Plot. What is the used degrees of freedom?

print("------------Q4---------------\n")
from sklearn.model_selection import LeaveOneOut
from pygam import LinearGAM
from pygam import s as s_gam

x = data['Girth'].to_numpy()
y = data['Volume'].to_numpy()
errors = []
para_list = np.linspace(0.1, 50, 50)

for para in para_list:
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    residual = []
    for i, (tr_idx, te_idx) in enumerate(loo.split(x)):
        gam = LinearGAM(s_gam(0, lam=para)).fit(x[tr_idx], y[tr_idx])
        residual.append(np.abs(gam.predict(x[te_idx]) - y[te_idx]))
    errors.append(np.mean(residual))

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('errors vs. smoothing para')
plt.scatter(para_list, errors,s=5)
plt.plot(para_list, errors,color='crimson')
plt.grid()
plt.xlabel('smoothing para')
plt.ylabel('error')

plt.subplot(122)
plt.title('smoothing spline')
x_range   = np.linspace(min(x), max(x), 100)
best_para = para_list[np.argmin(errors)]
gam       = LinearGAM(s_gam(0, lam=best_para)).fit(x, y)
y_pred    = gam.predict(x_range)
plt.scatter(x, y, label='data points')
plt.plot(x_range, y_pred, color='crimson',label='smoothing spline')
plt.grid()
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.legend()
plt.savefig('Problem2Q4.jpg',dpi=400)

#print(lam_list[np.argmin(all_errors)])
#print(degrees_of_freedom(x, gam.terms[0]))

print(f'smoothing para: {best_para:.2f}')
print(f"degree of freedom: {gam.statistics_['edof']:.2f}")

# %%
# Q5
# Use variable Girth and Height to predict the Volume by a GAM
# smoothing spline with df=4 (Girth) smoothing spline with df=5(Height)
# Plot with the condence bands
print("------------Q5---------------\n")

from pygam import LinearGAM, s

gam = LinearGAM(s(0, n_splines=4) + s(1, n_splines=5)).fit(data[['Girth', 'Height']], data['Volume'])
# Girth
XX = gam.generate_X_grid(term=0)
preds = gam.predict(XX)
confidence_intervals = gam.confidence_intervals(XX, width=0.95)
plt.figure(figsize=(10, 6))
plt.plot(XX[:, 0], preds, color='crimson', label='Prediction')
plt.fill_between(XX[:, 0], confidence_intervals[:, 0], confidence_intervals[:, 1], alpha=0.2,
                 label='95% Confidence Interval', color='pink')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.title('GAM Prediction and Confidence Intervals for Girth')
plt.legend()
plt.grid('on')
plt.savefig('Problem2Q5(girth_func).jpg', dpi=400)

# Height
XX = gam.generate_X_grid(term=1)
preds = gam.predict(XX)
confidence_intervals = gam.confidence_intervals(XX, width=0.95)
plt.figure(figsize=(10, 6))
plt.plot(XX[:, 1], preds, label='Prediction', color='crimson')
plt.fill_between(XX[:, 1], confidence_intervals[:, 0], confidence_intervals[:, 1], alpha=0.2,
                 label='95% Confidence Interval', color='pink')
plt.xlabel('Height')
plt.ylabel('Volume')
plt.title('GAM Prediction and Confidence Intervals for Height')
plt.legend()
plt.grid('on')
plt.savefig('Problem2Q5(height_func).jpg', dpi=400)