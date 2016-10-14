Nick Stevens
nps35@case.edu
10/9/2016

Artificial Neural Network

This is a matrix-based implementation of an ANN with one hidden layer and one output unit. It uses stochastic gradient
descent to update the weights during backpropagation. The learning rate is fixed at 0.01.

The program must be provided with the following five arguments:
    1. The name of the root directory of the dataset (e.g. "spam")
    2. 0 if using 5-fold stratified cross-validation, 1 if training and validating on the full sample
    3. The (integer) number of hidden units
    4. The (float) value of the weight decay coefficient
    5. The (integer) number of training iterations. Choose 0 to train until convergence.