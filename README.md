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

<img width="500" height="400" alt="Image" src="https://github.com/user-attachments/assets/993ba373-4b0a-45cc-99f3-4db8e0326255" />

 **The effect of the regularization parameter,C**

 Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns non-linear SVMs. Use scikit-learn's SVC function to learn SVM models with radial-basis kernels for fixed  γ  and various choices of  C∈{10−3,10−2⋯,1,⋯105} . The value of  γ  is fixed to  γ=1d⋅σX , where  d  is the data dimension and  σX  is the standard deviation of the data set  X . SVC can automatically use these setting for  γ  if you pass the argument gamma = 'scale' (see documentation for more details).

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Discussion: How do the training error and the validation error change with  C ? Based on the visualization of the models and their resulting classifiers, how does changing  C  change the models? Explain in terms of minimizing the SVM's objective function  12w′w+CΣni=1ℓ(w∣xi,yi) , where  ℓ  is the hinge loss for each training example  (xi,yi) .

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  Cbest . Report the accuracy on the test set for this selected best SVM model. Note: You should report a single number, your final test set accuracy on the model corresponding to  Cbest .

```py
# Learn support vector classifiers with a radial-basis function kernel with
# fixed gamma = 1 / (n_features * X.std()) and different values of C
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# ***Your Code Starts Here***

# Use np.arange to select a range from -3.0 to 6.0 with steps of 1.0
C_range = np.arange(-3, 6, 1)
# Use np.power to select C_values as 10.0 ^ C_range
C_values = np.power(10.0, C_range)

# ***Your Code Ends Here***

models = dict()
trnErr = dict()
valErr = dict()
gamma1=dict()


# ***Your Code Starts Here***

# Run the loop for all C_values
for C in C_values:
    #Crete a non-linear SVM classifier
    clf= SVC(kernel='rbf', gamma='scale', C= C)
    #Train Classifier
    models[C]= clf.fit(X_trn, y_trn)
    trnErr[C]= 1 - clf.score(X_trn, y_trn)
    valErr[C]= 1 - clf.score(X_val, y_val)

# ***Your Code Ends Here***


#visualise on training and validation data data
visualize(models, 'C', X_trn, y_trn)
# Plot all the models
plt.figure()
plt.semilogx(trnErr.keys(), trnErr.values(), marker='o', linewidth=3, markersize=12)
plt.semilogx(valErr.keys(), valErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('C', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(trnErr.keys()), fontsize=12)
plt.legend(['Training Error', 'Validation Error'], fontsize=16)

plt.axis([10**-3, 10**5, 0, 1])

# Find Best C and find the accuracy
C_best, Min_Error = min(valErr.items(), key=lambda x: x[1])
print('The best value of C is ', C_best)
clf=SVC(C = C_best, gamma='scale')
models=clf.fit(X_trn, y_trn)
Accuracy = (clf.score(X_tst, y_tst, sample_weight=None))
print('The accuracy of the model is ', Accuracy)

```

<img width="500" height="300" alt="Image" src="https://github.com/user-attachments/assets/1fe71b3a-380b-4267-86fe-5c35979ad0cc" />

## Breast Cancer Diagnosis with Support Vector Machines

 we will use the [Wisconsin Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) data set, which has already been pre-processed and partitioned into training, validation and test sets. Numpy's loadtxt command can be used to load CSV files.

## Breast Cancer Diagnosis with  k -Nearest Neighbors

 ```py
 # Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()

file_trn = '/content/wdbc_trn.csv'
file_val = '/content/wdbc_tst.csv'
file_tst = '/content/wdbc_val.csv'

X_trn=np.loadtxt(file_trn, usecols=np.arange(1, 31), delimiter = ",")
y_trn=np.loadtxt(file_trn, usecols=0, delimiter = ",")


X_val=np.loadtxt(file_val, usecols=np.arange(1, 31), delimiter = ",")
y_val=np.loadtxt(file_val, usecols=0, delimiter = ",")


X_tst=np.loadtxt(file_tst, usecols=np.arange(1, 31), delimiter = ",")
y_tst=np.loadtxt(file_tst, usecols=0, delimiter = ",")

```

```py
from sklearn.neighbors import KNeighborsClassifier
models = dict()
trnErr = dict()
valErr = dict()

# ***Your Code Starts Here***

k_values = np.arange(1,21, 3)

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_trn, y_trn)

    y_pred_train = neigh.predict(X_trn)
    y_pred_val = neigh.predict(X_val)
    models[k] =  neigh
    trnErr[k] = accuracy_score(y_trn, y_pred_train)
    valErr[k] = accuracy_score(y_val, y_pred_val)


# ***Your Code Ends Here***
# Plot all the models
plt.figure()
plt.plot(trnErr.keys(), trnErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(valErr.keys(), valErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('k', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(trnErr.keys()), fontsize=12)
plt.legend(['Training Error', 'Validation Error'], fontsize=16)


# ***Your Code Starts Here***

# Find Best gamma and find the accuracy
k_best, Min_Error = min(valErr.items(), key=lambda x: x[1])

# ***Your Code Ends Here***

print('The best value of K is ', k_best)
neigh = KNeighborsClassifier(n_neighbors=k_best)
models=neigh.fit(X_trn, y_trn)
Accuracy = (neigh.score(X_tst, y_tst, sample_weight=None))
print('The accuracy of the model is ', Accuracy)

```

<img width="747" height="533" alt="Image" src="https://github.com/user-attachments/assets/61c26607-d2ed-4c45-90b5-e36daef1141f" />

### Decision Trees with Synthetic Data

we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the generalization behavior of decision trees of increasing complexity, characterized by their depth,  d .

```py
   #
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_circles(n_samples=n_samples, noise=0.25, random_state=42, factor=0.3)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30)  # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], 0.25*np.eye(2), (m, ))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

```
```py
#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
def visualize(models, X, y):
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
        r, c = np.divmod(i, 3)
        if nrows == 1:
            ax = axes[c]
        else:
            ax = axes[r, c]

        # Plot contours
        zMesh = clf.predict(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('Tree Depth = {0}'.format(p))
```

<img width="720" height="510" alt="Image" src="https://github.com/user-attachments/assets/f3e960f6-ced8-40e6-9053-f9eeffbf07c5" />

## a. Model Selection and Visualization

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns decision trees. Use scikit-learn's DecisionTreeClassifier function to learn decision trees of different depths,  d∈{1,⋯,9} .

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly. Visualize the decision trees of different depths using the provided function.

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  dbest . Report the accuracy on the test set for this selected best decision tree model. Note: You should report a single number, your final test set accuracy on the model corresponding to  dbest .

```py
# Learn decision trees with different depths
from sklearn.tree import DecisionTreeClassifier

d_values = np.arange(1, 10, dtype='int')
models = dict()
trnErr = dict()
valErr = dict()

for d in d_values:
    #
    # INSERT YOUR CODE HERE
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = d)
    models[d] = model
    model.fit(X_trn, y_trn)
    #y_pred = model.predict(X_trn)
    trnErr[d] = model.score(X_trn,y_trn)
    #y_val_pred =
    valErr[d] = model.score(X_val,y_val)
    #
    print('Learning a decision tree with d = {0}.'.format(d))

visualize(models, X_trn, y_trn)
```
<img width="797" height="533" alt="Image" src="https://github.com/user-attachments/assets/70bda75c-8d89-4c6a-8c43-6c87304eda0c" />

```py
plt.figure(figsize=(10, 6))
plt.plot(d_values, list(trnErr.values()), label='Training Accuracy', marker='o')
plt.plot(d_values, list(valErr.values()), label='Validation Accuracy', marker='o')
plt.xlabel('Depth of the Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Depth of the Decision Tree')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="780" height="515" alt="Image" src="https://github.com/user-attachments/assets/146d03d4-ea78-4c5a-97ce-d7583728fcfa" />

## Model selection

**Cross validation:** Here, instead of a single validation set, we will use a  10 -fold cross validation procedure to improve robustness of model selection. Use scikit-learn's DecisionTreeClassifier function to perform model selection with decision trees of different depths,  d=3,⋯,15}  via  10 -fold cross validation. Make sure you are using entropy as the split criterion.

## Model Visualization

```py
#
# After you install GraphViz, EDIT THE CODE BELOW TO set the path to the
# executable 'dot.exe' below (shown for Windows). On MacOs, you can find GraphViz
# in the /Applications/ folder.
#
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\ete\\Desktop\\windows_10_cmake_Release_Graphviz-12.1.1-win32'

#
# DO NOT EDIT CODE BELOW
#
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

export_graphviz(models[d], out_file='tree.dot', feature_names=feature_names,
                class_names = ['loss', 'win'],
                rounded=True, proportion=False, precision=2, filled=True)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=150'])
Image(filename='tree.png')
```

<img width="852" height="227" alt="Image" src="https://github.com/user-attachments/assets/45e43192-e404-42fc-a9c2-9f01bf6e2a7d" />