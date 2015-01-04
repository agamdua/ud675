import pylab as pl
import numpy as np


class PlotGraph(object):
    """
    helper class for plotting graphs
    """
    def __init__(self, train_error, test_error, figure_number=None, k_range=None):
        self.train_error = train_error
        self.test_error = test_error

        assert k_range is not None
        self.k_range = k_range

        if not figure_number:
            self.figure_number = 0
        else:
            self.figure_number = figure_number

    def plot_train_test(self, title=None, xlabel='X axis', ylabel='MS Error'):
        """
        Plot the result of training and testing
        #TODO: not sure if this works, don't care for now
        #TODO: make this generic
        """
        # I would rather this be raised as not implemented than deal with
        # incorrect analysis
        raise NotImplementedError

        # Below should work, but cant say for sure.
        assert xlabel is not None
        assert ylabel is not None

        pl.figure(self.figure_number)
        self.figure_number = self.figure_number + 1

        pl.title(title)
        pl.plot(self.k_range, self.test_error, lw=2, label='test error')
        pl.plot(self.k_range, self.train_error, lw=2, label='training error')
        pl.legend()
        pl.xlabel(self.xlabel)
        pl.ylabel(self.ylabel)

    def plot_prediction_error(self, title='Prediction Error from util', xlabel='X axis', ylabel='MS Error', display_info=False):
        assert all([xlabel, ylabel])

        pl.figure(self.figure_number)

        pl.title(title)
        pl.plot(
            self.k_range, self.test_error, lw=2, label='variance'
        )
        pl.plot(
            self.k_range, np.square(self.train_error), lw=2, label='bias squared'
        )
        pl.plot(
            self.k_range, self.prediction_error, lw=2, label='prediction error'
        )
        pl.legend()
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)

        if display_info:
            self.display_info()

    def display_info(self):
        print "Minimum prediction error coordinates: {}".format(
            self.minimum_prediction_error_coordinates
        )
        print "Variance at this point: {}".format(
            self.variance_at_min_error
        )
        print "Bias at this point: {}".format(
            self.bias_at_min_error
        )

    @property
    def prediction_error(self):
        return np.square(self.test_error) + self.train_error

    @property
    def minimum_prediction_error(self):
        return min(self.prediction_error)

    @property
    def minimum_prediction_error_coordinates(self):
        return (
            np.where(self.prediction_error==self.minimum_prediction_error)[0][0],
            self.minimum_prediction_error
        )

    @property
    def variance_at_min_error(self):
        return self.test_error[self.minimum_prediction_error_coordinates[0]]

    @property
    def bias_at_min_error(self):
        return self.train_error[self.minimum_prediction_error_coordinates[0]]

    @classmethod
    def render(cls):
        """
        Convenience wrapper so that this can be called from this helper class

        Syntactic sugar, thats all
        """
        pl.show()
