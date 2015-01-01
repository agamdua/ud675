"""
Plots Learning curves for Neural Networks
Plot performance of NN when we change the size of the training set
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from neuralnets import NN

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the training set so that we have 10 different sizes
sizes = linspace(1, len(X_train), 10)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i,s in enumerate(sizes):
    # Build a network with 3 hidden layers
    net = NN(13, 9, 7, 3, 1)

    # Train the NN for 50 epochs
    # The .train() function returns MSE over the training set
    s = int(s)
    train_err[i] = net.train(X_train[:s], y_train[:s], num_epochs=50, verbose=False)

    # Find labels for the test set
    y = zeros(len(X_test))
    #print X_test
    for j in range(len(X_test)):
        y[j] = net.activate(X_test[j])

    # Find MSE for the test set
    #print 'Here is y:' + str(y)
    test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Neural Networks: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('MS Error')
pl.show()
