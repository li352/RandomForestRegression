import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:,[2]].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=50, random_state=0)
rfr.fit_transform(X,y)
y_pred= rfr.predict(6.5)
y_pred

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='blue')
plt.plot(X_grid, rfr.predict(X_grid), color='green')
plt.title('RFR')
plt.xlabel('positionsalaries')
plt.ylabel('salary')
plt.show() 