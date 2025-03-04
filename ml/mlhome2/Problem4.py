import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


'-----------------Preporcessing------------------------'
# 
data = pd.read_csv('titanic.csv')
X    = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']] # feature
y    = data['Survived']                                # label
mask = X.notna().all(axis=1)                           # mask the NaN
X    = X[mask]                                                                          
y    = y[mask]                                         # drop NaN from X and y

X    = pd.get_dummies(X, columns=['Sex'])              # convert 'sex' to two columns with 0, 1
X    = pd.get_dummies(X, columns=['Pclass'])
X    = X.astype('float')                               # ensure the whole X is numerical type
#%%
# Q1
print('-----------Q1----------\n')
X = X[['Pclass_3','Sex_male','Age','SibSp','Fare','Pclass_1']]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
result = logit_model.fit()
#print(result.summary())
print(f'coefficient:\n{result.params}')
conf_intervals = result.conf_int()
conf_intervals.columns = ['2.5%', '97.5%']
print("95% confidence intervals:")
print(conf_intervals.loc[['Sex_male', 'Pclass_3']])

#%%
# Q2 Bootstrap
print('-----------Q2----------\n')
n = 1000
bootstrapped_coefs = []

for _ in range(n):
    X_resampled, y_resampled = resample(X, y)
    logit_model_resampled = sm.Logit(y_resampled, X_resampled)
    result_resampled = logit_model_resampled.fit(disp=0)
    bootstrapped_coefs.append(result_resampled.params)

bootstrapped_coefs = pd.DataFrame(bootstrapped_coefs)
bootstrapped_conf_intervals = bootstrapped_coefs.quantile([0.025, 0.975])
print("Bootstrap 95% confidence intervals:")
print(bootstrapped_conf_intervals[['Sex_male', 'Pclass_3']])


#%%
# Q4 predict the survive porbability of the first point
print('-----------Q4----------\n')
X_train = X.iloc[1:]
y_train = y.iloc[1:]
X_test = X.iloc[0:1]

logit_model_train = sm.Logit(y_train, X_train)
result_train = logit_model_train.fit()
predicted_prob = result_train.predict(X_test)
print(f"Predicted probability of survival for the test point: {predicted_prob[0]}\n")

bootstrap_preds = []
for _ in range(n):
    X_resampled, y_resampled = resample(X_train, y_train)
    logit_model_resampled = sm.Logit(y_resampled, X_resampled)
    result_resampled = logit_model_resampled.fit(disp=0)
    bootstrap_preds.append(result_resampled.predict(X_test))

bootstrap_preds = np.array(bootstrap_preds)
prediction_interval = np.percentile(bootstrap_preds, [2.5, 97.5])
print(f"Bootstrap 95% prediction interval for the test point's survival probability: {prediction_interval}\n")

#%%
# Q5 do the same thing in Q4 with QDA
print('-----------Q5----------\n')
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train.drop(columns='const'), y_train)  # Drop the constant column for QDA
predicted_prob_qda = qda.predict_proba(X_test.drop(columns='const'))
print(f"Predicted probability of survival for the test point using QDA: {predicted_prob_qda[0,1]}\n")

bootstrap_preds_qda = []

for _ in range(n):
    X_resampled, y_resampled = resample(X_train.drop(columns='const'), y_train)
    qda.fit(X_resampled, y_resampled)
    proba_group = qda.predict_proba(X_test.drop(columns='const'))
    bootstrap_preds_qda.append(proba_group[0,1])

bootstrap_preds_qda = np.array(bootstrap_preds_qda)
prediction_interval_qda = np.percentile(bootstrap_preds_qda, [2.5, 97.5])
print(f"Bootstrap 95% prediction interval for the test point's survival probability using QDA: {prediction_interval_qda}")

#%%
# Q3 Explore the dataset as you like and report some of your findings

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(2, 2, figsize=(15, 10))


sns.countplot(x='Survived', data=data, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Survival')
axs[0, 0].set_xlabel('Survival (0 = No, 1 = Yes)')
axs[0, 0].set_ylabel('Count')

sns.countplot(x='Sex', hue='Survived', data=data, ax=axs[0, 1])
axs[0, 1].set_title('Survival Rate by Sex')
axs[0, 1].set_xlabel('Sex')
axs[0, 1].set_ylabel('Count')
axs[0, 1].legend(title='Survival', loc='upper right', labels=['No', 'Yes'])

sns.countplot(x='Pclass', hue='Survived', data=data, ax=axs[1, 0])
axs[1, 0].set_title('Survival Rate by Passenger Class')
axs[1, 0].set_xlabel('Passenger Class')
axs[1, 0].set_ylabel('Count')
axs[1, 0].legend(title='Survival', loc='upper right', labels=['No', 'Yes'])

sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Age Distribution by Survival')
axs[1, 1].set_xlabel('Age')
axs[1, 1].set_ylabel('Count')

plt.tight_layout()

plt.savefig('P4Q3.jpg',dpi=400)

