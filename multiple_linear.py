import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
load = pd.read_csv('FuelConsumption.csv')
main = load
separate = main[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
separator =  np.random.rand(len(separate)) < 0.8
train = main[separator]
test = main[~separator]
from sklearn import linear_model
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
regression.fit(train_x, train_y)
print(f"Coefficients is: {regression.coef_}")
print(f"Intercept is: {regression.intercept_}")
train_y_pred = regression.predict(train_x)
test_y_pred = regression.predict(test_x)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("\n--- Train Metrics ---")
print("Mean Absolute Error (MAE): %.2f" % mean_absolute_error(train_y, train_y_pred))
print("Mean Squared Error (MSE): %.2f" % mean_squared_error(train_y, train_y_pred))
print("R2-score: %.2f" % r2_score(train_y, train_y_pred))
print("\n--- Test Metrics ---")
print("Mean Absolute Error (MAE): %.2f" % mean_absolute_error(test_y, test_y_pred))
print("Mean Squared Error (MSE): %.2f" % mean_squared_error(test_y, test_y_pred))
print("R2-score: %.2f" % r2_score(test_y, test_y_pred))
plt.figure(figsize =(10,6))
plt.scatter(train_y, train_y_pred, color = 'blue', alpha = 0.4, label = 'Train Data')
plt.scatter(test_y, test_y_pred, color = 'red', alpha = 0.4, label = 'Test Data')
plt.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()],
color = 'green', linestyle = '--', linewidth = 2, label = 'Perfect Prediction')
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()
