import random
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import KDTree

def randomSelect(train, test, n, **kwargs):
    """
    Randomly selects 'n' data points from the test set and moves them to the train set.
    Args:
        train (np.array): Current training dataset.
        test (np.array): Current testing dataset.
        n (int): Number of data points to be moved.
    Returns:
        tuple: Updated training and testing datasets.
    """
    for _ in range(n):
        # select random point and move it to train set
        data_idx = random.randint(0, len(test)-1)
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)
    return train, test

def densitySelect(train, test, n, utility, beta):
    """
    Selects 'n' points from the test set based on their density-related utility to the model.
    Args:
        train (np.array): Training set.
        test (np.array): Test set.
        n (int): Number of data points to move.
        utility (float): Utility factor for density calculation.
        beta (float): Exponent for adjusting influence of density in utility calculation.
    Returns:
        tuple: Updated training and testing datasets.
    """
    for _ in range(n):
        # length of test data
        r_len = len(test)

        # instantiate similarity and information arrays
        sim_arr = np.zeros(shape=(r_len,))
        info_arr = np.zeros(shape=(r_len,))

        # calculate similarities for each unlabeled point to all other unlabeled points
        for i in range(r_len):
                for j in range(i+1):
                        if i != j:
                            #similarity is 1/distance
                            sim = (1 / euclidean(test[i], test[j]))
                            sim_arr[i] += sim
                            sim_arr[j] += sim
                # calculate information from normalized similarity
                sim_arr[i] = ((1 / r_len) * sim_arr[i])
                info_arr[i] = utility * sim_arr[i]**beta
        
        # select argmax informative point and move it to train set
        data_idx = np.argmax(info_arr)
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)
            
    return train, test

def uncertaintyKNNSelect(train, test, n):
    """
    Selects 'n' points from the test set based on uncertainty of classification using a KNN classifier.
    
    This function identifies the test points where the KNN classifier's prediction is most uncertain, 
    indicating a near equal probability of belonging to either class, and moves them to the training set.
    
    Args:
        train (np.array): The training dataset.
        test (np.array): The test dataset.
        n (int): Number of data points to move.

    Returns:
        tuple: Updated training and testing datasets.
    """
    k = 5  # number of nearest neighbors to consider

    for _ in range(n):
        # create a KD-Tree from the training dataset for efficient neighbor search
        kd_tree = KDTree(train)
        # query the tree for the k nearest neighbors of each point in the test dataset
        _, indices = kd_tree.query(test, k=k)

        # initialize an array to store the predicted probabilities
        predict_proba = np.zeros((test.shape[0],))

        # calculate the probability predictions for the test data
        for i in range(len(indices)):
            # accumulate votes from the k-nearest neighbors
            for j in range(len(indices[i])):
                predict_proba[i] += train[indices[i][j], 2] - 1  # assuming the third column in train data stores class probabilities
            # adjust the probabilities to reflect uncertainty
            predict_proba_adj = np.abs(predict_proba[i]/k - 0.5)  # Normalize and measure closeness to 0.5 (maximum uncertainty)

        # identify the point with the minimum adjusted probability (highest uncertainty)
        min = np.min(predict_proba_adj)
        idxs = np.asarray(np.where(predict_proba_adj == min))

        # handle ties randomly if multiple indices have the same minimum value
        if len(idxs) > 1:
            data_idx = idxs[0, random.randint(0, len(idxs[0]))]
        else:
            data_idx = idxs[0, 0]

        # move the most uncertain point from the test set to the training set
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)

    return train, test

def qbcKNNSelect1(train, test, n, c_size):
    """
    Implements query-by-committee to select data points from the test set to add to the train set based on committee disagreement.
    Args:
        train (np.array): Training dataset.
        test (np.array): Test dataset.
        n (int): Number of data points to move.
        c_size (int): Number of committees.
    Returns:
        tuple: Updated training and testing datasets.
    """
    # separate target and feature array
    test_X, test_y = np.delete(test, 2, axis=1), test[:,2]
    # instantiate predicted probability array
    predict_proba = np.zeros((len(test_y)),)

    for _ in range(n):
        k = 5
        # split training set into c_size committees
        c_members = np.array_split(train, c_size)
        # if min committee size - 1 is smaller than k, lower k
        min_len = np.min([np.shape(x)[0] for x in c_members])
        if min_len - 1 < k:
            k = min_len - 1
        # train KNN on each committee
        for member in c_members:
            # separte target and feature array
            train_X, train_y = np.delete(member, 2, axis=1), member[:,2]
            model = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
            # count number of class 1 predictions
            predict_proba += model.predict(test_X) - 1
        predict_proba_adj = np.abs(predict_proba / c_size - 0.5)
        # select most conflicted data point by comittee vote and move to train set
        min = np.min(predict_proba_adj)
        idxs = np.asarray(np.where(predict_proba_adj==min))
        if len(idxs) > 1:
            data_idx = idxs[0,random.randint(0, len(idxs[0]))]
        else:
            data_idx = idxs[0,0]
        
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)
    return train,test

def qbcKNNSelect2(train, test, n, min_k, max_k):
    """
    Selects 'n' points from the test set to add to the training set using a Query by Committee (QBC) strategy.
    This function employs multiple KNN classifiers with varying 'k' values to form a committee. The point
    where the committee shows the highest disagreement (uncertainty) is selected to move to the training set.
    
    Args:
        train (np.array): Training dataset, assumed to have the last column as the target variable.
        test (np.array): Test dataset, similarly structured as the training dataset.
        n (int): Number of data points to select and move from test to train.
        min_k (int): Minimum value of 'k' for KNN classifiers in the committee.
        max_k (int): Maximum value of 'k' for KNN classifiers in the committee.
        
    Returns:
        tuple: Updated training and testing datasets after moving 'n' points from test to train.
    """
    # Separate features and target variable for the test set
    test_X, test_y = np.delete(test, 2, axis=1), test[:,2]
    # Initialize array to accumulate predicted probabilities
    predict_proba = np.zeros((len(test_y)),)

    # Adjust max_k if it is larger than the number of available examples in the training set
    if len(train) <= max_k:
        max_k = len(train) - 1  # Ensure max_k is less than the number of training samples
    c_size = max_k - min_k  # Define committee size based on the range of 'k'
    
    for _ in range(n):
        for i in range(c_size):
            k = min_k + i  # incrementally use different 'k' values for the committee members
            # separate features and target for the training set
            train_X, train_y = np.delete(train, 2, axis=1), train[:,2]
            # train a KNN model with 'k' neighbors and predict on the test set
            model = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
            # accumulate predictions from each committee member
            predict_proba += model.predict(test_X) - 1
        # normalize and calculate absolute uncertainty for committee's predictions
        predict_proba_adj = np.abs(predict_proba / c_size - 0.5)
        
        # find the test point with minimum adjusted probability, indicating maximum uncertainty
        min = np.min(predict_proba_adj)
        idxs = np.asarray(np.where(predict_proba_adj==min))
        # resolve ties randomly if multiple points have the same minimum uncertainty
        if len(idxs) > 1:
            data_idx = idxs[0, random.randint(0, len(idxs[0]))]
        else:
            data_idx = idxs[0, 0]

        # move the most uncertain point from the test to the training set
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)

    return train, test


def qbcKNNSelect1Mellow(train, test, n, c_size):
    """
    Similar to qbcKNNSelect but selects points more conservatively based on disagreement.
    Args:
        train (np.array): Training dataset.
        test (np.array): Test dataset.
        n (int): Number of data points to move.
        c_size (int): Number of committees.
    Returns:
        tuple: Updated training and testing datasets.
    """
    # separate target and feature array
    test_X, test_y = np.delete(test, 2, axis=1), test[:,2]

    for _ in range(n):
        k = 5
        # split training set into c_size committees
        c_members = np.array_split(train, c_size)
        # if min committee size - 1 is smaller than k, lower k
        min_len = np.min([np.shape(x)[0] for x in c_members])
        disagree_lst = []
        if min_len - 1 < k:
            k = min_len - 1
        # train KNN on each committee
        for i, member in enumerate(c_members):
            # separte target and feature array
            train_X, train_y = np.delete(member, 2, axis=1), member[:,2]
            model = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
            predictions = model.predict(test_X)
            # store committee 1 predictions
            if i == 0:
                predict_arr = predictions
            # if committee disagrees, index for potential selection
            else:
                for j, pred in enumerate(predict_arr):
                    if pred != predictions[j]:
                        disagree_lst.append(j)
        # randomly select any conflicted data point by comittee vote and move to train set, if no disagreement select randomly
        if len(disagree_lst) > 0:
            data_idx = random.choice(disagree_lst)
        else:
            data_idx = random.randint(0, len(test))
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)
    return train,test

def uncertaintyRFSelect(train, test, n, predict_proba):
    """
    Selects data points from the test set to move to the train set based on uncertainty in Random Forest predictions.
    
    This function uses the smallest difference between the top two predicted probabilities as a measure of uncertainty.
    
    Args:
        train (np.array): The training dataset.
        test (np.array): The test dataset.
        n (int): The number of data points to move from the test set to the training set.
        predict_proba (np.array): Array of predicted probabilities for each class for each test instance.
    
    Returns:
        tuple: Updated training and testing datasets.
    """
    for _ in range(n):
        # calculate the absolute difference between the two highest predicted probabilities
        pred_proba_adj = np.abs(predict_proba[0]-predict_proba[1])
        # find the index of the test example with the minimum difference (highest uncertainty)
        data_idx = np.argmin(pred_proba_adj)
        # add the most uncertain test example to the training set
        train = np.concatenate((train, np.array([test[data_idx]])), axis=0)
        # remove the moved example from the test set
        test = np.concatenate((test[:data_idx], test[data_idx+1:]), axis=0)
    return train, test

def qbcRFSelect(train, test, n, predict_proba):
    """
    A wrapper function that calls the uncertainty-based selection function for Random Forest classifiers.
    
    Essentially, this function provides an interface that matches the uncertainty selection criteria using
    Random Forest prediction probabilities.
    
    Args:
        train (np.array): The training dataset.
        test (np.array): The test dataset.
        n (int): The number of data points to move from the test set to the training set.
        predict_proba (np.array): Array of predicted probabilities for each class for each test instance.
    
    Returns:
        tuple: Updated training and testing datasets.
    """
    return uncertaintyRFSelect(train, test, n, predict_proba)
