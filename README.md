### Regression 

# Generating Synthetic Data
This assignment shows how we can extend ordinary least squares regression, which uses the hypothesis class of linear regression functions, to non-linear regression functions modeled using polynomial basis functions and radial basis functions. The function we want to fit is  ytrue=ftrue(x)=6sin(x+2)+sin(2x+4) . This is a univariate function as it has only one input variable. First, we generate synthetic input (data)  xi  by sampling  n=750  points from a uniform distribution on the interval  [−7.5,7.5] .

```py
# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y
  import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

```

```py
import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')
 ```

 <img width="697" height="507" alt="Image" src="https://github.com/user-attachments/assets/bf6884a4-8bc6-4ed4-a754-b1c42d757a17" />
 
Recall that we want to build a model to generalize well on future data, and in order to generalize well on future data, we need to pick a model that trade-off well between fit and complexity (that is, bias and variance). We randomly split the overall data set ( D ) into three subsets:

- **Training set:**  Dtrn  consists of the actual training examples that will be used to train the model

- **Validation set:**  Dval  consists of validation examples that will be used to tune model hyperparameters (such as  λ>0  in ridge regression) in order to find the best trade-off between fit and complexity (that is, the value of  λ  that produces the best model);
-Test set:  Dtst  consists of test examples to estimate how the model will perform on future data.

<img width="833" height="195" alt="Image" src="https://github.com/user-attachments/assets/3d89d87c-830e-49cc-97cc-fc68359b0fe6" />

```py
# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
```

<img width="715" height="538" alt="Image" src="https://github.com/user-attachments/assets/4b67ec9d-b11c-4810-afd5-0e37c471bafa" />

# Regression with Polynomial Basis Function

This problem extends ordinary least squares regression, which uses the hypothesis class of linear regression functions, to non-linear regression functions modeled using polynomial basis functions. In order to learn nonlinear models using linear regression, we have to explicitly transform the data into a higher-dimensional space. The nonlinear hypothesis class we will consider is the set of  d -degree polynomials of the form  f(x)=w0+w1x+w2x2+...+wdxd  or a linear combination of polynomial basis function:

```py
# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
  #
  #
  # *** Insert your code here ***
  #convert the array into numpy
  
    #X = np.asarray(X)

  # Create the Vandermonde matrix for the input data up to degree d
    V = np.zeros((len(X), d + 1))
    
    # Loop over each degree and fill in the columns with powers of X
    for i in range(len(X)):
        for j in range(d+1):
            V[i,j]=np.power(i,j)
    return V
```

```py
# Check your Polynomial Function

Phi_trn = polynomial_transform(X_trn, 3)
Phi_val = polynomial_transform(X_val, 3)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='yellow')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
```
<img width="300" height="250" alt="Image" src="https://github.com/user-attachments/assets/cb7a8197-d397-4deb-b12a-942e67e3804a" />

## Support Vector Machine

we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the behavior of SVMs with Radial-Basis Function (RBF) kernels with different values of  C  and  γ .


```py
#
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
  # Generate a non-linear data set
  X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

  # Take a small subset of the data and make it VERY noisy; that is, generate outliers
  m = 30
  np.random.seed(42)
  ind = np.random.permutation(n_samples)[:m]
  X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
  y[ind] = 1 - y[ind]

  # Plot this data
  cmap = ListedColormap(['#b30065', '#178000'])
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

  # First, we use train_test_split to partition (X, y) into training and test sets
  X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,
                                                random_state=42)

  # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
  X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac,
                                                random_state=42)

  return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

```

```py
#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#

def visualize(models, param, X, y):
  # Initialize plotting
  if len(models) % 3 == 0:
    nrows = len(models) // 3
  else:
    nrows = len(models) // 3 + 1

  fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
  cmap = ListedColormap(['#b30065', '#178000'])

  # Create a mesh
  xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
  yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),
                             np.arange(yMin, yMax, 0.01))

  for i, (p, clf) in enumerate(models.items()):
    # if i > 0:
    #   break
    r, c = np.divmod(i, 3)
    ax = axes[r, c]

    # Plot contours
    zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

    if (param == 'C' and p > 0.0) or (param == 'gamma'):
      ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1],
                 alpha=0.5, linestyles=['--', '-', '--'])

    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
    ax.set_title('{0} = {1}'.format(param, p))
```

```py
# Generate the data
n_samples = 300    # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)


```

<img width="300" height="250" alt="Image" src="https://github.com/user-attachments/assets/993ba373-4b0a-45cc-99f3-4db8e0326255" />