#	7.	Deep Learning

## Artificial Neural Network (ANN)

![ANN](/7DL/Assets/ANN.png)

### Techniques used is stochastic gradient descent
-	Supply the network with bunck of samples
-	Every time the network predict a wrong result, the weight is adjusted 

![SGD](/7DL/Assets/SGD.png)

`Keras -> Library used to create neural network`

```python 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
```

## 	Convolutional Neural Network (CNN)

### 