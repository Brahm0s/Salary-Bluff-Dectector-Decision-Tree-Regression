# Descision Tree Regression

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Descision Treee Regresiion to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# Prediction of the result
yPred = regressor.predict(6.5)

# Visualising the Descision Tree Regression result
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff{Linear Regression}')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree Regression result(Higher Resolution)
xGrid = np.arange(min(x), max(x), 0.01)
xGrid = xGrid.reshape((len(xGrid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff{Linear Regression}')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()