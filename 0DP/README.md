#  0.	Data Preprocessing

## Importing Dataset

### Import the data in form of CSV into our environment
-	Pandas library will take care of the task

```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values
```

### Splitting the data into test and train set
-	Neccessary to make sure that we can train the model then test it.
-	It has to be in the right proportion 

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

`train_test_split -> to split all the data into train and test set`

### Feature Scaling
-	Make sure that the number is properly scaled between -1 and 1 for easy processing

```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
```
`StandardScaler -> to the scale the data within the range`


## Missing Data

![Missing Data](/0DP/Assets/Missing.png)

### There are two strategies to solve the missing data 
-	Remove the particular record or dont use it
-	Replace the missing data with the mean
- 	Need to import Imputer from sklearn.preprocessing

`Imputer  -> Compute the missing values`
-	missing_values = 'NaN'
-	stategy = 'mean' or median or most_frequent
-	axis = 0

```python
Imputer = Imputer.fit(X[:, 1:3])  # 1:3 Upperbound is excluded
X[:, 1:3] = Imputer.transform(X[:, 1:3]) # need to be transformed again
```


## Categorical Data 

### Encode the variable into number so that it can be understood by computer
-	Need to trasform the data which are non-number into number
-	Need to import LabelEncoder from sklearn.preprocessing

`LabelEncoder -> Transform non-numerical data into numerical data`

![Dummy](/0DP/Assets/Dummy.png)

### There is a problem using LabelEncoder, the label given shouldnt be used for comparison
-	Need to do Dummy Encoding
-	Need to import OneHotEncoder from sklearn.preprocessing

`OneHotEncoder -> Encode categorical integer features using one-of-K scheme`
-	categorical_features = [0]