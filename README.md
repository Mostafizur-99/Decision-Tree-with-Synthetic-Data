### Regression 

# Generating Synthetic Data
This assignment shows how we can extend ordinary least squares regression, which uses the hypothesis class of linear regression functions, to non-linear regression functions modeled using polynomial basis functions and radial basis functions. The function we want to fit is  ytrue=ftrue(x)=6sin(x+2)+sin(2x+4) . This is a univariate function as it has only one input variable. First, we generate synthetic input (data)  xi  by sampling  n=750  points from a uniform distribution on the interval  [−7.5,7.5] .