#	8.	Dimension Reduction

## Principle Component Analysis (PCA)

![PCA](/8DR/Assets/PCA.png)

### Reducing the dimension of samples by looking at the variability of feature samples, steps:
-	Subtract the mean of data from each variables
-	Calculate and form a covariance matrix
-	Calculate eigenvectors and eigenvalues from the covariance matrix
-	Choose feature vectors 
-	Multiply the transposed feature vectors bytransposed adjusted data

`PCA -> library used for PCA`

```python
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
```

## Kernel PCA

### Similar to PCA section but using kernel

`KernelPCA -> used in this section`

```python
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
```

## Linear Discriminant Analysis (LDA)

![LDA](/8DR/Assets/LDA.png)

### LDA can be used for dimension reduction and classification method
-	Maximize the variability of samples

`LinearDiscriminantAnalysis -> library for LDA`

```python
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
```
