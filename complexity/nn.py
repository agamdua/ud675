"""
Plots Performance of Neural Networks when you change the network
We vary complexity by changing the number of hidden layers the network has
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

# List all the different networks we want to test again
# All networks have 13 input nodes and 1 output nodes
# All networks are fully connected
# No hidden layers
net0 = NN(13,1)
# 1 hidden layer with 5 nodes
net1 = NN(13,5,1)
# 2 hidden layers with 7 and 3 nodes resp
net2 = NN(13,7,3,1)
# 3 hidden layers with 9, 7 and 3 nodes resp
net3 = NN(13,9,7,3,1)
# 4 hidden layers with 9, 7, 3 and 2 nodes resp
net4 = NN(13,9,7,3,2,1)
# 5 hidden layers with 10, 8, 4, 3, 2 nodes resp
net5 = NN(13,10,8,4,3,2,1)

# We will train each network on the dataset
nets = [net0, net1, net2, net3, net4, net5]
num_hidden = [0,1,2,3,4,5]

train_err = zeros(len(nets))
test_err = zeros(len(nets))

# We will train each NN for 50 epochs
max_epochs = 50

for i, net in enumerate(nets):
	# Run backprop for max_epochs number of times
	train_err[i] = net.train(X_train, y_train, num_epochs=max_epochs, verbose=False)

	# Find the labels for test set
	y = zeros(len(X_test))

	for j in range(len(X_test)):
		y[j] = net.activate(X_test[j])

    # Calculate MSE for all samples in the test set
	test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the number of hidden layers
pl.figure()
pl.title('Neural Networks: Performance vs Model Complexity')
pl.plot(num_hidden, test_err, lw=2, label = 'test error')
pl.plot(num_hidden, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Hidden Layers')
pl.ylabel('MS Error')
pl.show()
