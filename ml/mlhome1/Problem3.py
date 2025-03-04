import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# loading data
train = pd.read_csv('boston_housing_train.csv')
test  = pd.read_csv('boston_housing_test.csv')

# split x and y
x_train = np.array(train.drop('medv', axis=1)) # drop y row by row
y_train = np.array(train['medv'])
x_test  = np.array(test.drop('medv', axis=1))
y_test  = np.array(test['medv'])

#%% standardize the X
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s  = scaler.transform(x_test)

#%%
def get_distance(vect1, vect2):  # calculate the Euclidean distance between two vectors
    vect = vect1 - vect2
    dis  = 0
    for i in vect:
        dis = dis + i**2
    dis  = np.sqrt(dis)
    return dis

def KNNreg(train_x, train_y, test_x, k):
    predictions  = []
    
    for testvec in test_x:
        distance = []
        
        for trainvec in train_x:
            dis  = get_distance(trainvec, testvec)
            distance.append(dis)                      # caculate the distance between every row vector
                                                      # from train_x and the vector from test we want 
                                                      # to predict its y

        kneighbor_index = np.argsort(distance)[:k]    # find K-neigbor index
        kneighbor       = train_y[kneighbor_index]    # get the data according to index
        
        predictions.append(kneighbor.mean())          # get the mean from neighbor data as our prediction
    return predictions

#%%
def evaluate(train_x, train_y, test_x, k):
    start_time  = time.time()
    predictions = KNNreg(train_x, train_y, test_x, k)  # time to run the KNN
    end_time    = time.time()
    
    return predictions, end_time - start_time

#%%

k_range = np.arange(1,100)
MeanSE     = []
Time    = []
pred    = []
for k in k_range:
    predvalue, t = evaluate(x_train, y_train, x_test, k)
    mse = mean_squared_error(y_test, predvalue)
    MeanSE.append(mse)
    Time.append(t)
    pred.append(predvalue)

plt.figure(figsize=[11,5])
plt.subplot(121)
plt.plot(k_range, MeanSE, 'r',label='MSE')
plt.xlabel('k')
plt.ylabel('MSE')
plt.legend()
plt.subplot(122)
plt.plot(k_range, Time, 'b',label ='time')
plt.xlabel('k')
plt.ylabel('Time(s)')
plt.legend()
plt.savefig('original.jpg',dpi=400)
#%%
# If standardize:
k_range = np.arange(1,100)
MeanSE     = []
Time    = []
pred    = []
for k in k_range:
    predvalue, t = evaluate(x_train_s, y_train, x_test_s, k)
    mse = mean_squared_error(y_test, predvalue)
    MeanSE.append(mse)
    Time.append(t)
    pred.append(predvalue)

plt.figure(figsize=[11,5])
plt.subplot(121)
plt.plot(k_range, MeanSE, 'r',label='MSE')
plt.xlabel('k')
plt.ylabel('MSE')
plt.legend()
plt.subplot(122)
plt.plot(k_range, Time, 'b',label ='time')
plt.xlabel('k')
plt.ylabel('Time(s)')
plt.legend()
plt.savefig('standardize.jpg',dpi=400)