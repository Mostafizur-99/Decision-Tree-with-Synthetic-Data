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

-Training set:  Dtrn  consists of the actual training examples that will be used to train the model
-Validation set:  Dval  consists of validation examples that will be used to tune model hyperparameters (such as  λ>0  in ridge regression) in order to find the best trade-off between fit and complexity (that is, the value of  λ  that produces the best model);
-Test set:  Dtst  consists of test examples to estimate how the model will perform on future data.