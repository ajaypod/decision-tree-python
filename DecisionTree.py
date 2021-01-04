import datetime as dt
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class DecisionTree:
    """
    Note: Current methodology is that for a Decision Tree Regressor. 
    Future work will include is that for a classifier specifically 
    redefining the information gain formula.
    Methods:
        - __init__ -> initialize the variables you think you will need
        - train_tree -> takes in two numpy arrays as inputs (inputs, outputs)
                        and creates a tree graph using an information gain formula of your choosing.
                        You should also be able to tune the leaf node size of the tree.
        - row_predict -> takes in a numpy 1-d array returns a prediction as a float
        - predict -> takes in a numpy array and returns an array of predictions

    """

    def __init__(self,decision_boundary=list(),node_feature=list()):
        """Class initialization

        Args:
            decision_boundary (list, optional): List to capture the node bounds. Defaults to list().
            node_feature (list, optional): List to capture feature based on which node is split. Defaults to list().
        """        
        self.left = None
        self.right = None
        self.prediction = None
        self.decision_boundary = decision_boundary
        self.node_feature = node_feature


    def train_tree(self, inputs, outputs, min_leaf_size=10):
        """
        inputs - a numpy array
        outputs - a numpy array
        min_leaf_size - an integer determining the minimum number of records
                        a node needs in order to be further split.
        """

        self.min_leaf_size = min_leaf_size

        assert outputs.ndim == 1, "Error: y variable array needs to be 1d"
        # Initiae best_split_feat to record best split feature
        best_split_feat = 0
        # Record Variance (Reduction in Variance is used as the decision tree splitting method)
        min_variance = np.var(outputs)
        # Capture no.of features in the input array
        inp_shape = inputs.shape[-1]
        
        if len(inputs) <= min_leaf_size:
            self.prediction = np.mean(outputs)
            return

        for i in range(inp_shape):
            
            var_left_len = outputs[inputs[:, i]<np.mean(inputs[:, i])].size
            var_right_len = outputs[inputs[:, i]>=np.mean(inputs[:, i])].size

            if var_left_len==0 or var_right_len==0:
                self.prediction = np.mean(outputs)
                return

            var_left = np.var(outputs[inputs[:, i]<np.mean(inputs[:, i])])
            var_right = np.var(outputs[inputs[:, i]>=np.mean(inputs[:, i])])
            
            # Calculate weighted variance for each split
            var = (var_right*var_right_len + var_left*var_left_len)/(var_right_len+var_left_len)

            # Record the split with the least variance
            if var<min_variance:
                min_variance = var
                best_split_feat = i
                
        # Traverse down the tree
        if best_split_feat!=0:
            left_X = inputs[inputs[:, best_split_feat]<np.mean(inputs[:, best_split_feat])]
            right_X = inputs[inputs[:, best_split_feat]>=np.mean(inputs[:, best_split_feat])]
            left_y = outputs[inputs[:, best_split_feat]<np.mean(inputs[:, best_split_feat])]
            right_y = outputs[inputs[:, best_split_feat]>=np.mean(inputs[:, best_split_feat])]    

            self.decision_boundary.append(np.mean(inputs[:, best_split_feat]))
            self.node_feature.append(best_split_feat)
            self.left = DecisionTree(self.decision_boundary,self.node_feature)
            self.right = DecisionTree(self.decision_boundary,self.node_feature)

            self.left.train_tree(inputs = left_X, outputs = left_y,min_leaf_size=self.min_leaf_size)
            self.right.train_tree(inputs = right_X,outputs = right_y,min_leaf_size=self.min_leaf_size)
        else:
            self.prediction = np.mean(outputs)

        return

    def row_predict(self, xinput,j=0):
        """Generate predictions by iterating over each row of input array

        Args:
            xinput (numpy array): numpy 1-d array
            j (int, optional): increment counter to traverse the tree. Defaults to 0.

        Returns:
            numpy float: prediction - y for input array
        """
        if self.prediction is not None:
            return self.prediction
        elif self.left or self.right is not None:
            if xinput[self.node_feature[j]]>=self.decision_boundary[j]:
                return self.right.row_predict(xinput,j=j+1)
            else:
                return self.left.row_predict(xinput,j=j+1)

    def predict(self, inputs):
        """
        This function takes a numpy array, and given the decision tree structure
        created by the train_tree method, creates a new numpy array of predictions
        and returns those predictions
        Args:
            inputs - a numpy array
        Returns:
            numpy array: numpy array of predictions
        """
        y_predict = list()
        for row in range(inputs.shape[0]):
            y_predict.append(self.row_predict(inputs[row,:],j=0))
        
        return np.array(y_predict)


def test_model(data, random_state, min_leaf_size):
    """
    Test the DecisionTree class on multiple parameters
    with multiple inputs.
    """
    # split the data into train and test subsets
    # with the random state set, the output will be the same every time
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1],
                                                        data[:, -1],
                                                        test_size=0.2,
                                                        random_state=random_state)
    # Start the timer
    start = dt.datetime.now()
    # Create the model
    tree_model = DecisionTree()
    # train the model
    tree_model.train_tree(x_train, y_train, min_leaf_size=min_leaf_size)
    # create predictions
    predictions = tree_model.predict(x_test)
    # record speed time
    speed = (dt.datetime.now() - start).total_seconds()
    #record root mean squared error
    rmse = round(sqrt(mean_squared_error(y_test, predictions)), 2)

    return [random_state, min_leaf_size, rmse, speed, predictions]

