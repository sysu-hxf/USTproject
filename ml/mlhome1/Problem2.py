#%% Data Preprocessing
import pandas as pd
import statsmodels.api as sm
import numpy as np
data = pd.read_csv('Life Expectancy Data.csv')
#print(data.info())
new_status = []                          # Status is supposed to convert to numerical type
for i in data['Status']:
     if i is not np.nan:
         new_status.append(10 - len(i))  # developing -> 0 ; developed -> 1
     else:
         new_status.append(np.nan)       # nan -> nan
data['Status'] = pd.Series(new_status)   
data = data.dropna()                    # delete the row if it contains nan
#print(data.info())
#%% Q1 
'''
    Summary of the linear model. 
    Find predicting variables actually affecting the life. 
'''
#  drop Country and Life expectancy, X1 is consisted of all covariates
X1 = data.drop([data.columns[0],data.columns[3]], axis=1)
X1 = sm.add_constant(X1)                 # add constant term to X1
Y = data['Life expectancy ']             # Y is Life expectancy as response

model1 = sm.OLS(Y,X1).fit()
print(model1.summary())


#%% Q2
'''
    Construct the 95% confidence intervals for the coefficient of Adult Mortality & HIV/AIDS.
    Do these predictors definitely have negative impact on the life expectancy?
'''
X2 = data[['Adult Mortality', ' HIV/AIDS']]  
X2 = sm.add_constant(X2)
model2 = sm.OLS(Y, X2).fit()
conf_int_2 = model2.conf_int(alpha=0.05)  # 95% confidence intervals

print(model2.summary())
print(f"95% Confidence Intervals:\n{conf_int_2}")


#%% Q3
'''
    Construct the 97% confidence intervals for the coefficient of Schooling & Alcohol. 
    Explain how these predictors impact the life expectancy.
'''
X3 = data[['Schooling', 'Alcohol']]  
X3 = sm.add_constant(X3)
model3 = sm.OLS(Y, X3).fit()
conf_int_3 = model3.conf_int(alpha=0.03)  # 97% confidence intervals

print(model3.summary())
print(f"97% Confidence Intervals:\n{conf_int_3}")


#%% Q4
'''
    The top-seven most influential predictors.
    Use these predictors to fit a smaller model and report the summary
'''
# select 7 predictors from model1
p_values = model1.pvalues.sort_values()
top_var = p_values.index[:6]
top_var = top_var.append(p_values.index[7:8]) # 排除截距项
X_top = sm.add_constant(data[top_var])
new_model = sm.OLS(Y,X_top).fit()
print(new_model.summary())


#%% Q5
'''
Year=2008, Status=Developed, Adult Mortality=125, infant deaths=94, Alcohol=4.1, percentage expenditure=100, Hepatitis B=20, Measles=13, BMI=55, under-five deaths=2, Polio=12, Total expenditure=5.9, Diphtheria=12,
HIV/AIDS=0.5, GDP=5892,Population=1.34 × 106, Income composition of resources=0.9, Schooling=18.

Report the 99% confidence interval for your prediction
'''

test = pd.DataFrame({
    'const': 1,  
    ' HIV/AIDS': [0.5],
    'Adult Mortality': [125],
    'Schooling':[18],
    'Income composition of resources':[0.9],
    'under-five deaths': [2],
    'infant deaths': [94],
    'Year':[2008]
})


pred = new_model.get_prediction(test)
result = pred.summary_frame(alpha=0.01)

print("Life Expectancy:", result.iloc[0]['mean'])
print("99% Confidence Interval:\n", result.iloc[0]['mean_ci_lower'], ' ~ ',result.iloc[0]['mean_ci_upper'])

#%% Q6

model1_aic = model1.aic
new_model_aic = new_model.aic
# get AIC
print(f"Full Model AIC: {model1_aic}")
print(f"Small Model AIC: {new_model_aic}")