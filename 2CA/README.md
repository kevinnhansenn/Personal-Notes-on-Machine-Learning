#	2.	Classification Algorithm

## Logistic Regression

![Logistic](/2CA/Assets/Logistic.png)

### Statistical model that use logistic function to model binary dependent variable 

`LogisticRegression -> to create logistic model`

```python
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

### The folllowing code is for visualization

```python
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## KNN

![KNN](/2CA/Assets/KNN.png)

### Classifying the new sample depending to n closest samples
-	If there are more samples in category A then B within the closest n samples,
-	Then the new sample belogns to category A

`KNeighborsClassifer -> for KNN`

```python
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

## SVM

![SVM](/2CA/Assets/SVM.png)

### Draw the decision boundaries using the mean of both classes and two closest samples from different category

`SVC -> library for SVM`

```python
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

## Kernel SVM

![Kernel](/2CA/Assets/Kernel.png)

### Using kernel to map the feature into higher dimension
-	Sometime the samples are not linearly separable
-	We need to use kernel trick so that they may be separable in higher dimension

```python
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
```

## Naïve Bayes

![Naive](/2CA/Assets/Naive.png)

### It is categrorized into simple probablistic classifier
-	Simplest amongst Bayesian network

`GaussinaNB -> library used for naive bayes`

```python
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

## Decision Tree Classification

![Tree](/2CA/Assets/Tree.png)

### This part is very similar to decision tree regression

`DecisionTreeClassifier -> is used for classification algroithm`

```python
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

## Random Forest Classification

![Forest](/2CA/Assets/Forest.png)

### This part is very similar to Random Forest Regression part 

`RandomForestClassifer -> is used in this case`

```python
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
```
