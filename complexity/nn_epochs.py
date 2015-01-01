"""
Plots Performance of Neural Networks with number of epochs
This is not for studying model complexity, but to see how long does it take for NN to converge
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

# We will test if the NN converges in 200 iterations
max_epochs = 200
train_err = zeros(max_epochs)
test_err = zeros(max_epochs)

# Build a network with 13 input nodes, 5 hidden nodes and 1 output nodes
# The networks is fully connected - a node from a layer is connected to all nodes 
# in its neighboring layer
net = NN(13, 5, 1)

for i in range(max_epochs):
	# Run the backprop once
    train_err[i] = net.train(X_train, y_train, num_epochs=1, verbose=False)

    # Find the labels for test set
    y = zeros(len(X_test))
    for j in range(0, len(X_test)):
    	# Run X_test[j] on the NN and determine its label
        y[j] = net.activate(X_test[j])

    # Calculate MSE for all samples in the test set
    test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the number of epochs (iterations)
pl.figure()
pl.title('Neural Networks: Performance vs Num of Epochs')
pl.plot(range(max_epochs), test_err, lw=2, label = 'test error')
pl.plot(range(max_epochs), train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Epochs')
pl.ylabel('MS Error')
pl.show()
