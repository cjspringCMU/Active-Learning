import random
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from selectASR import densitySelect, qbcRFSelect, randomSelect, qbcKNNSelect1, qbcKNNSelect2, qbcKNNSelect1Mellow

def generateKwargs(query, rf_output=None):
        """
        Generate keyword arguments for different selection functions based on the query type.
    
        Args:
            query (function): The selection function to customize arguments for.
            rf_output (np.array, optional): Output of a RandomForest model if applicable.

        Returns:
            dict: A dictionary of keyword arguments specific to the given query type.
        """
        kwargs = dict()
        if query == densitySelect:
            kwargs['beta'] = 1
            kwargs['utility'] = 1
        elif query == qbcKNNSelect1 or query == qbcKNNSelect1Mellow:
            kwargs['c_size'] = 4
        elif query == qbcKNNSelect2:
            kwargs['min_k'] = 3
            kwargs['max_k'] = 5
        elif query == qbcRFSelect:
            kwargs['predict_proba'] = rf_output
        return kwargs
    
def simulateModel(classifier, query, data, seed, starting_data=10):
    """
    Simulate a model training process using a query-based selection strategy.
    
    Args:
        classifier: Machine learning classifier to be used.
        query (function): Query strategy for selecting new data points.
        data (np.array): Dataset to be used in the simulation.
        seed (int): Seed value for random operations to ensure reproducibility.
        starting_data (int): Number of initial data points to start training.

    Returns:
        tuple: Arrays containing the count of training data and test accuracies over rounds.
    """
    # set seed
    round = 0
    random.seed(seed)

    # initialize train and test sets
    train = np.empty([0,3])
    test = data

    # initialize training count and mse arrays
    train_cnts = np.empty([0,1])
    test_acc = np.empty([0,1])

    kwargs = dict()

    # run until test and train have 50 data points each
    while len(test) > 50:
        args = []
        # split data
        if round == 0:  
            train, test = randomSelect(train, test, starting_data)
        else:
            train, test = query(train, test, 1, **kwargs)

        # separate target and feature array
        train_X, train_y = np.delete(train, 2, axis=1), train[:,2]
        test_X, test_y = np.delete(test, 2, axis=1), test[:,2]

        # fit and predict on test data
        model = classifier.fit(train_X, train_y)
        pred_y = model.predict(test_X)

        # store training counts and mse
        train_cnts = np.append(train_cnts, len(train))
        test_acc = np.append(test_acc, accuracy_score(test_y, pred_y))

        if isinstance(classifier, RandomForestClassifier):
            args.append(model.predict_proba(test_X))
        kwargs = generateKwargs(query, *args)

        round += 1
    return train_cnts, test_acc

def simulateModels(n, data, classifier, query, seeds):
    """
    Simulate multiple model training processes and average results.
    
    Args:
        n (int): Number of simulations to run.
        data (np.array): The dataset to use for all simulations.
        classifier: The classifier to use in the simulations.
        query (function): The query strategy for data selection.
        seeds (list of int): Seed values for each simulation to ensure reproducibility.

    Returns:
        tuple: Arrays of training counts, mean test accuracies, and their standard deviations across simulations.
    """
    for i in range(n):
        # start simulation with all data in test, empty train
        train = np.empty([0,3])
        test = data

        # run one simulation and store acc
        train_cnts, test = simulateModel(classifier, query, data, seeds[i])
        if i == 0:
            test_acc = test
        else:
            test_acc = np.vstack((test_acc, test))

    # calculate mean and std across simulations for each round
    test_acc_means = np.mean(test_acc, axis=0)
    test_acc_stds = np.std(test_acc, axis=0)

    return train_cnts, test_acc_means, test_acc_stds
