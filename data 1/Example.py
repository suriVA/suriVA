import pandas as pd
import sklearn as sk  

df = pd.read_csv('data 1/TEST DATA1.csv')
y= df("logs")
x= df.drop('Logs', axis=1)

#Splitting data into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=5)

#Training Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

#Apply model
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

#Evaluate model Performance
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(Y_train, y_lr_train_pred)
lr_train_r2 = r2_score(Y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(Y_test, y_lr_test_pred)
lr_test_r2 = r2_score(Y_test, y_lr_test_pred)

lr_results= pd.DataFrame(('Liner Regression', lr_train_mse, lr_train_r2, lr_test_r2)).transpose()
lr_results.columns =["Method", "Trainig MSE", "Training R2", "Test MSE", "Test R2"]

#Random Forest Liner Regression model
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(max_depth=2, random_state=5)
rf.fit(X_train, Y_train)

#Apply Model
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

#Evaluate Model
rf_train_mse = mean_squared_error(Y_train, y_rf_train_pred)
rf_train_r2 = r2_score(Y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(Y_test, y_rf_test_pred)
rf_test_r2 = r2_score(Y_test, y_rf_test_pred)

rf_results= pd.DataFrame(('Liner Regression', rf_train_mse, rf_train_r2, rf_test_r2)).transpose()
rf_results.columns =["Method", "Trainig MSE", "Training R2", "Test MSE", "Test R2"]

#Combine Results
df_models = pd.concat([lr_results, rf_results], axis=1)
df_models.reset_index(drop=True)

#Visualtions of results
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(x=Y_train, y=y_rf_train_pred, alpha=0.3)
z= np.polyfit(Y_train, y_lr_train_pred, 1)
p= np.polyld(z)

plt.plot(Y_train, p(Y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
