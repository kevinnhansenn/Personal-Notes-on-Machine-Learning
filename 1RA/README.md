# 	1.	Regression Algorithm

## Simple Linear Regression 

![Linear Regression](/1RA/Assets/Linear_Regression.png)

### Determine relationship between data by drawing the linear best fit line 
-	Drawing a straight line with the minimum distance from all samples
-	Ordinary Least method 
-	Make a predition based on the best fit line

`LinearRegression -> to create regressor for liear regression`

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

## Multiple Linear Regression 

![Multiple Regression](/1RA/Assets/Multiple_Regression.png)

### Determine relationship between data by srawing hte polynomial line 
-	Drawing the best fit line with polynomial function 
-	Need to determine the degree of polynomial

```python
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
```

## Polynomial Regression 

### Similar to linear regression, but instead of using linear function, we use non-linear function with degree

-	Using polynomial function for prediction

`PolynomialFeatures -> for doing polynomial regression`

```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```

## Support Vector Regression 

![SVR](/1RA/Assets/SVR.png)

### Fitting as many instances as possible while limiting margin violation
-	Can be linear and nonlinear functin
-	Using kernel: map lower dimension data to hiher dimension data

`SVR -> for doing SVR regression`

```python
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
y_pred = sc_y.inverse_transform(y_pred)
```

## Decision Tree Regression 

![Tree](/1RA/Assets/Tree.png)

### Performing the split based on the entropy
-	The algorithm will keep splitting until it can a certain condition is met
-	Information entropy

`DecisionTreeRegressor -> for decision tree regression`

```python
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
```

## Random Forest Regression 

### It is an extension from decision tree regression
### Steps:
-	Pick K data points from training sets
-	Build decision tree using K data points
-	Choose number of trees we want to build and repeat step 1 and 2
-	For new data point, make each of the trees predict and average all the reults

`RandomForestRegressor -> for creating random forest regression`

```python
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
```