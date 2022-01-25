#!/usr/bin/env python
# coding: utf-8

# ## Jason Brown
# ## CS 458 Solar Prediction
# ## Nov 24

# Below are all the imports used throughout the project

# In[1]:


import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


# In[2]:


# Read from file, split into list
df = pd.read_csv("solar.csv")
df


# In[ ]:


df = pd.read_csv("Solar.csv") 
# rename vars
df=df.rename(columns ={
    'ZONEID':'ZONEID',
    'TIMESTAMP':'TIMESTAMP',
    'VAR78':'liquid_water',
    'VAR79':'ice_water',
    'VAR134':'surface_pressure',
    'VAR157':'relative_humidity',
    'VAR164':'cloud_cover',
    'VAR165':'u_wind_component',
    'VAR166':'v_wind_component',
    'VAR167':'two_m_temp',
    'VAR169':'solar_rad_down',
    'VAR175':'thermal_rad_down',
    'VAR178':'net_solar_rad',
    'VAR228':'total_precip',
    'CURR_POWER':'POWER'
})

# + 24 hours
powerplus24 = df.iloc[:,-1]
powerplus24 = powerplus24.shift(periods = -24) 
powerplus24 = powerplus24.iloc[:-24]
df = df.iloc[:-24, :]

# create new column for 24 hour ahead predictions in the dataframe
df['24_hours'] = powerplus24
training_df = df[df['TIMESTAMP'] <= '20130701 00:00']
testing_df = df[df['TIMESTAMP'] >= '20130701 00:00']

training_df.to_csv('solar_training.csv') # index = False
testing_df.to_csv('solar_testing.csv') # index = False
df.head(5)


# Below is a chart that represents the correlations between all the variables in the dataframe.  The column of interest is the 24ahead column that represents affects variables have on the next day's power generation.

# In[ ]:


# display dataframe correlations
corr = df.corr()
corr.style.background_gradient(cmap = 'binary')


# Split the data into zones and do some preprocessing

# In[ ]:


#print(df_train.columns)
#Dropping timestamp for scaling, but save timestamp columns for later use in analysis
Z1_times = df_test[(df_test['ZONEID']== 1)]
Z1_times = Z1_times['TIMESTAMP']


Z2_times = df_test[(df_test['ZONEID']== 2)]
Z2_times = Z2_times['TIMESTAMP']

Z3_times = df_test[(df_test['ZONEID']== 3)]
Z3_times = Z3_times['TIMESTAMP']


df_train.drop('TIMESTAMP', 1, inplace = True)
df_test.drop('TIMESTAMP', 1, inplace = True)


# In[ ]:


#Split training and test DataFrames into zones
Z1_train = df_train[(df_train['ZONEID']== 1)]
Z2_train = df_train[(df_train['ZONEID']== 2)]
Z3_train = df_train[(df_train['ZONEID']== 3)]

Z1_test = df_test[(df_test['ZONEID']== 1)]
Z2_test = df_test[(df_test['ZONEID']== 2)]
Z3_test = df_test[(df_test['ZONEID']== 3)]

#Seperate X and Y values
X_train1 = Z1_train.iloc[:, :-1]
X_train2 = Z2_train.iloc[:, :-1]
X_train3 = Z3_train.iloc[:, :-1]

y_train1 = Z1_train.iloc[:, -1]
y_train2 = Z2_train.iloc[:, -1]
y_train3 = Z3_train.iloc[:, -1]

X_test1 = Z1_test.iloc[:, :-1]
X_test2 = Z2_test.iloc[:, :-1]
X_test3 = Z3_test.iloc[:, :-1]

y_test1 = Z1_test.iloc[:, -1]
y_test2 = Z2_test.iloc[:, -1]
y_test3 = Z3_test.iloc[:, -1]


# In[ ]:


#Get rid of Zone IDs now that data is seperated
X_train1.drop('ZONEID', 1, inplace = True)
X_test1.drop('ZONEID', 1, inplace = True)

X_train2.drop('ZONEID', 1, inplace = True)
X_test2.drop('ZONEID', 1, inplace = True)

X_train3.drop('ZONEID', 1, inplace = True)
X_test3.drop('ZONEID', 1, inplace = True)

#Scale the data
scaler = MinMaxScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.fit_transform(X_test1)

X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.fit_transform(X_test2)

X_train3 = scaler.fit_transform(X_train3)
X_test3 = scaler.fit_transform(X_test3)


# In[ ]:


#Zone 1
#Create MLPRegressor and fit data
reg = MLPRegressor(activation = 'logistic', solver = 'sgd')
reg.fit(X_train1 ,y_train1)
#Get Prediction
y_pred1 = reg.predict(X_test1)

#Store prediction, actual, and timestamps into dataframe
mlp_result1 = pd.DataFrame({'TIMESTAMP': Z1_times, 'Actual': y_test1, 'Prediction': y_pred1})

#Mean Absolute error
MAE1 = mean_absolute_error(y_test1, y_pred1)
print("Zone 1 Mean Absolute Error:", MAE1)
#Root mean squared error
RMSE1 = mean_squared_error(y_test1, y_pred1)
print("Zone 1 Root Mean Squared Error:", RMSE1, '\n')


#Zone2
#Create MLPRegressor and fit data
reg = MLPRegressor(activation = 'logistic', solver = 'sgd')
reg.fit(X_train2 ,y_train2)
y_pred2 = reg.predict(X_test2)

#Store prediction, actual, and timestamps into dataframe
mlp_result2 = pd.DataFrame({'TIMESTAMP': Z2_times, 'Actual': y_test2, 'Prediction': y_pred2})


#Mean Absolute error
MAE2 = mean_absolute_error(y_test2, y_pred2)
print("Zone 2 Mean Absolute Error:", MAE2)
#Root mean squared error
RMSE2 = mean_squared_error(y_test2, y_pred2)
print("Zone 2 Root Mean Squared Error:", RMSE2, '\n')



#Zone 3
#Create MLPRegressor and fit data
reg = MLPRegressor(activation = 'logistic', solver = 'sgd')
reg.fit(X_train3 ,y_train3)
y_pred3 = reg.predict(X_test3)

#Store prediction, actual, and timestamps into dataframe
mlp_result3 = pd.DataFrame({'TIMESTAMP': Z3_times, 'Actual': y_test3, 'Prediction': y_pred3})


#Mean Absolute error
MAE3 = mean_absolute_error(y_test3, y_pred3)
print("Zone 3 Mean Absolute Error:", MAE3)
#Root mean squared error
RMSE3 = mean_squared_error(y_test3, y_pred3)
print("Zone 3 Root Mean Squared Error:", RMSE3, '\n')

#Get averages of all zones
avgMAE = (MAE1 + MAE2 + MAE3)/3
avgRMSE = (RMSE1 + RMSE2 + RMSE3)/3
print("Average MAE:", avgMAE)
print("Average RMSE", avgRMSE)


# In[ ]:


# #Potential parameters to use
# kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# shrinking = ['True', 'False']
# for kern in kernel:
#     for shrink in shrinking:
#         reg.fit(X_train1 ,y_train1)
#         y_pred1 = reg.predict(X_test1)
#         MAE = mean_absolute_error(y_test1, y_pred1)
#         print("For", clf, "MAE:", MAE)

#Zone 1
reg = SVR(kernel = 'linear')
reg.fit(X_train1 ,y_train1)
y_pred1 = reg.predict(X_test1)


svr_result1 = pd.DataFrame({'TIMESTAMP': Z1_times, 'Actual': y_test1, 'Prediction': y_pred1})
#print(df_result1.head(20))

#Mean Absolute error
MAE1 = mean_absolute_error(y_test1, y_pred1)
print("Zone 1 Mean Absolute Error:", MAE1)
#Root mean squared error
RMSE1 = mean_squared_error(y_test1, y_pred1)
print("Zone 1 Root Mean Squared Error:", RMSE1, '\n')



#Zone2
reg = SVR(kernel = 'linear')
reg.fit(X_train2 ,y_train2)
y_pred2 = reg.predict(X_test2)

svr_result2 = pd.DataFrame({'TIMESTAMP': Z2_times, 'Actual': y_test2, 'Prediction': y_pred2})
#print(df_result2.head(20))

#Mean Absolute error
MAE2 = mean_absolute_error(y_test2, y_pred2)
print("Zone 2 Mean Absolute Error:", MAE2)
#Root mean squared error
RMSE2 = mean_squared_error(y_test2, y_pred2)
print("Zone 2 Root Mean Squared Error:", RMSE2, '\n')



#Zone 3
reg = SVR(kernel = 'linear')
reg.fit(X_train3 ,y_train3)
y_pred3 = reg.predict(X_test3)

svr_result3 = pd.DataFrame({'TIMESTAMP': Z3_times, 'Actual': y_test3, 'Prediction': y_pred3})
#print(df_result2.head(20)3)

#Mean Absolute error
MAE3 = mean_absolute_error(y_test3, y_pred3)
print("Zone 3 Mean Absolute Error:", MAE3)
#R3oot mean squared error
RMSE3 = mean_squared_error(y_test3, y_pred3)
print("Zone 3 Root Mean Squared Error:", RMSE3, '\n')
avgMAE = (MAE1 + MAE2 + MAE3)/3
avgRMSE = (RMSE1 + RMSE2 + RMSE3)/3
print("Average MAE:", avgMAE)
print("Average RMSE", avgRMSE)


# Analyze the predictions vs actual over small time periods to better visualize the results

# In[ ]:


mlpZ1_short = mlp_result1[mlp_result1['TIMESTAMP'] >= '20130928 00:00']
mlpZ1_short = mlpZ1_short[mlpZ1_short['TIMESTAMP'] <= '20130929 00:00']

mlpZ2_short = mlp_result2[mlp_result2['TIMESTAMP'] >= '20130928 00:00']
mlpZ2_short = mlpZ2_short[mlpZ2_short['TIMESTAMP'] <= '20130929 00:00']

mlpZ3_short = mlp_result3[mlp_result3['TIMESTAMP'] >= '20130928 00:00']
mlpZ3_short = mlpZ3_short[mlpZ3_short['TIMESTAMP'] <= '20130929 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,10))
fig.suptitle("Multilayer Perceptron Power Predictions 9/28/2013", fontsize = 20)
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Time', ylabel= 'Power')
ax1.legend()
x = range(0,24)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Time', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Time', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


svrZ1_short = svr_result1[svr_result1['TIMESTAMP'] >= '20130928 00:00']
svrZ1_short = svrZ1_short[svrZ1_short['TIMESTAMP'] <= '20130929 00:00']

svrZ2_short = svr_result2[svr_result2['TIMESTAMP'] >= '20130928 00:00']
svrZ2_short = svrZ2_short[svrZ2_short['TIMESTAMP'] <= '20130929 00:00']

svrZ3_short = svr_result3[svr_result3['TIMESTAMP'] >= '20130928 00:00']
svrZ3_short = svrZ3_short[svrZ3_short['TIMESTAMP'] <= '20130929 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Support Vector Power Predictions 9/28/2013", fontsize = 20)
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Time', ylabel= 'Power')
ax1.legend()
x = range(0,24)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Time', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Time', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


mlpZ1_short = mlp_result1[mlp_result1['TIMESTAMP'] >= '20140407 00:00']
mlpZ1_short = mlpZ1_short[mlpZ1_short['TIMESTAMP'] <= '20140414 00:00']

mlpZ2_short = mlp_result2[mlp_result2['TIMESTAMP'] >= '20140407 00:00']
mlpZ2_short = mlpZ2_short[mlpZ2_short['TIMESTAMP'] <= '20140414 00:00']

mlpZ3_short = mlp_result3[mlp_result3['TIMESTAMP'] >= '20140407 00:00']
mlpZ3_short = mlpZ3_short[mlpZ3_short['TIMESTAMP'] <= '20140414 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Multilayer Perceptron Power Predictions 04/07/2014 - 04/14/2014", fontsize = 20)
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Day', ylabel= 'Power')
ax1.legend()
x = range(7,14)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Day', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Day', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


svrZ1_short = svr_result1[svr_result1['TIMESTAMP'] >= '20140407 00:00']
svrZ1_short = svrZ1_short[svrZ1_short['TIMESTAMP'] <= '20140414 00:00']

svrZ2_short = svr_result2[svr_result2['TIMESTAMP'] >= '20140407 00:00']
svrZ2_short = svrZ2_short[svrZ2_short['TIMESTAMP'] <= '20140414 00:00']

svrZ3_short = svr_result3[svr_result3['TIMESTAMP'] >= '20140407 00:00']
svrZ3_short = svrZ3_short[svrZ3_short['TIMESTAMP'] <= '20140414 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Support Vector Power Predictions 04/07/2014 - 04/14/2014", fontsize = 20)
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Day', ylabel= 'Power')
ax1.legend()
x = range(7,14)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Day', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Day', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# Remove current power variable and rerun training to see impact on results

# In[ ]:


#Remove previous power from data to see difference in results
X_train1 = pd.DataFrame(X_train1)
X_train1 = X_train1.iloc[:,:-1]

X_test1 = pd.DataFrame(X_test1)
X_test1 = X_test1.iloc[:,:-1]

X_train2 = pd.DataFrame(X_train2)
X_train2 = X_train2.iloc[:,:-1]

X_test2 = pd.DataFrame(X_test2)
X_test2 = X_test2.iloc[:,:-1]

X_train3 = pd.DataFrame(X_train3)
X_train3 = X_train3.iloc[:,:-1]

X_test3 = pd.DataFrame(X_test3)
X_test3 = X_test3.iloc[:,:-1]


# In[ ]:


reg = MLPRegressor()
reg.fit(X_train1 ,y_train1)
y_pred1 = reg.predict(X_test1)

mlp_result1 = pd.DataFrame({'TIMESTAMP': Z1_times, 'Actual': y_test1, 'Prediction': y_pred1})


#Mean Absolute error
MAE1 = mean_absolute_error(y_test1, y_pred1)
print("Zone 1 Mean Absolute Error:", MAE1)
#Root mean squared error
RMSE1 = mean_squared_error(y_test1, y_pred1)
print("Zone 1 Root Mean Squared Error:", RMSE1, '\n')


#Zone2
reg = MLPRegressor()
reg.fit(X_train2 ,y_train2)
y_pred2 = reg.predict(X_test2)

mlp_result2 = pd.DataFrame({'TIMESTAMP': Z2_times, 'Actual': y_test2, 'Prediction': y_pred2})


#Mean Absolute error
MAE2 = mean_absolute_error(y_test2, y_pred2)
print("Zone 2 Mean Absolute Error:", MAE2)
#Root mean squared error
RMSE2 = mean_squared_error(y_test2, y_pred2)
print("Zone 2 Root Mean Squared Error:", RMSE2, '\n')



#Zone 3
reg = MLPRegressor()
reg.fit(X_train3 ,y_train3)
y_pred3 = reg.predict(X_test3)

mlp_result3 = pd.DataFrame({'TIMESTAMP': Z3_times, 'Actual': y_test3, 'Prediction': y_pred3})


#Mean Absolute error
MAE3 = mean_absolute_error(y_test3, y_pred3)
print("Zone 3 Mean Absolute Error:", MAE3)
#Root mean squared error
RMSE3 = mean_squared_error(y_test3, y_pred3)
print("Zone 3 Root Mean Squared Error:", RMSE3, '\n')

avgMAE = (MAE1 + MAE2 + MAE3)/3
avgRMSE = (RMSE1 + RMSE2 + RMSE3)/3
print("Average MAE:", avgMAE)
print("Average RMSE", avgRMSE)


# In[ ]:


mlpZ1_short = mlp_result1[mlp_result1['TIMESTAMP'] >= '20130928 00:00']
mlpZ1_short = mlpZ1_short[mlpZ1_short['TIMESTAMP'] <= '20130929 00:00']

mlpZ2_short = mlp_result2[mlp_result2['TIMESTAMP'] >= '20130928 00:00']
mlpZ2_short = mlpZ2_short[mlpZ2_short['TIMESTAMP'] <= '20130929 00:00']

mlpZ3_short = mlp_result3[mlp_result3['TIMESTAMP'] >= '20130928 00:00']
mlpZ3_short = mlpZ3_short[mlpZ3_short['TIMESTAMP'] <= '20130929 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,10))
fig.suptitle("Multilayer Perceptron Power Predictions Without Previous Power 9/28/2013", fontsize = 20)
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Time', ylabel= 'Power')
ax1.legend()
x = range(0,24)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Time', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Time', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


#Zone 1
reg = SVR()
reg.fit(X_train1 ,y_train1)
y_pred1 = reg.predict(X_test1)


svr_result1 = pd.DataFrame({'TIMESTAMP': Z1_times, 'Actual': y_test1, 'Prediction': y_pred1})
#print(df_result1.head(20))

#Mean Absolute error
MAE1 = mean_absolute_error(y_test1, y_pred1)
print("Zone 1 Mean Absolute Error:", MAE1)
#Root mean squared error
RMSE1 = mean_squared_error(y_test1, y_pred1)
print("Zone 1 Root Mean Squared Error:", RMSE1, '\n')



#Zone2
reg = SVR()
reg.fit(X_train2 ,y_train2)
y_pred2 = reg.predict(X_test2)

svr_result2 = pd.DataFrame({'TIMESTAMP': Z2_times, 'Actual': y_test2, 'Prediction': y_pred2})
#print(df_result2.head(20))

#Mean Absolute error
MAE2 = mean_absolute_error(y_test2, y_pred2)
print("Zone 2 Mean Absolute Error:", MAE2)
#Root mean squared error
RMSE2 = mean_squared_error(y_test2, y_pred2)
print("Zone 2 Root Mean Squared Error:", RMSE2, '\n')



#Zone 3
reg = SVR()
reg.fit(X_train3 ,y_train3)
y_pred3 = reg.predict(X_test3)

svr_result3 = pd.DataFrame({'TIMESTAMP': Z3_times, 'Actual': y_test3, 'Prediction': y_pred3})
#print(df_result2.head(20)3)

#Mean Absolute error
MAE3 = mean_absolute_error(y_test3, y_pred3)
print("Zone 3 Mean Absolute Error:", MAE3)
#R3oot mean squared error
RMSE3 = mean_squared_error(y_test3, y_pred3)
print("Zone 3 Root Mean Squared Error:", RMSE3, '\n')
avgMAE = (MAE1 + MAE2 + MAE3)/3
avgRMSE = (RMSE1 + RMSE2 + RMSE3)/3
print("Average MAE:", avgMAE)
print("Average RMSE", avgRMSE)


# In[ ]:


svrZ1_short = svr_result1[svr_result1['TIMESTAMP'] >= '20130928 00:00']
svrZ1_short = svrZ1_short[svrZ1_short['TIMESTAMP'] <= '20130929 00:00']

svrZ2_short = svr_result2[svr_result2['TIMESTAMP'] >= '20130928 00:00']
svrZ2_short = svrZ2_short[svrZ2_short['TIMESTAMP'] <= '20130929 00:00']

svrZ3_short = svr_result3[svr_result3['TIMESTAMP'] >= '20130928 00:00']
svrZ3_short = svrZ3_short[svrZ3_short['TIMESTAMP'] <= '20130929 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Support Vector Power Predictions No Previous Power 9/28/2013", fontsize = 20)
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Time', ylabel= 'Power')
ax1.legend()
x = range(0,24)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Time', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Time', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


mlpZ1_short = mlp_result1[mlp_result1['TIMESTAMP'] >= '20140407 00:00']
mlpZ1_short = mlpZ1_short[mlpZ1_short['TIMESTAMP'] <= '20140414 00:00']

mlpZ2_short = mlp_result2[mlp_result2['TIMESTAMP'] >= '20140407 00:00']
mlpZ2_short = mlpZ2_short[mlpZ2_short['TIMESTAMP'] <= '20140414 00:00']

mlpZ3_short = mlp_result3[mlp_result3['TIMESTAMP'] >= '20140407 00:00']
mlpZ3_short = mlpZ3_short[mlpZ3_short['TIMESTAMP'] <= '20140414 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Multilayer Perceptron Power Predictions No Previous Power 04/07/2014 - 04/14/2014", fontsize = 20)
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(mlpZ1_short['TIMESTAMP'], mlpZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Day', ylabel= 'Power')
ax1.legend()
x = range(7,14)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(mlpZ2_short['TIMESTAMP'], mlpZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Day', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(mlpZ3_short['TIMESTAMP'], mlpZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Day', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")


# In[ ]:


svrZ1_short = svr_result1[svr_result1['TIMESTAMP'] >= '20140407 00:00']
svrZ1_short = svrZ1_short[svrZ1_short['TIMESTAMP'] <= '20140414 00:00']

svrZ2_short = svr_result2[svr_result2['TIMESTAMP'] >= '20140407 00:00']
svrZ2_short = svrZ2_short[svrZ2_short['TIMESTAMP'] <= '20140414 00:00']

svrZ3_short = svr_result3[svr_result3['TIMESTAMP'] >= '20140407 00:00']
svrZ3_short = svrZ3_short[svrZ3_short['TIMESTAMP'] <= '20140414 00:00']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,10))
fig.suptitle("Support Vector Power Predictions No Previous Power 04/07/2014 - 04/14/2014", fontsize = 20)
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Actual'], c = 'black', label= 'Actual')
ax1.plot(svrZ1_short['TIMESTAMP'], svrZ1_short['Prediction'], c = 'r', label = 'Prediction')
ax1.set(xlabel = 'Day', ylabel= 'Power')
ax1.legend()
x = range(7,14)
ax1.set_xticklabels(x)
ax1.set_title("Zone 1")

ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Actual'], c = 'black', label= 'Actual')
ax2.plot(svrZ2_short['TIMESTAMP'], svrZ2_short['Prediction'], c = 'r', label = 'Prediction')
ax2.set(xlabel = 'Day', ylabel= 'Power')
ax2.legend()
ax2.set_xticklabels(x)
ax2.set_title("Zone 2")

ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Actual'], c = 'black', label= 'Actual')
ax3.plot(svrZ3_short['TIMESTAMP'], svrZ3_short['Prediction'], c = 'r', label = 'Prediction')
ax3.set(xlabel = 'Day', ylabel= 'Power')
ax3.legend()
ax3.set_xticklabels(x)
ax3.set_title("Zone 3")

