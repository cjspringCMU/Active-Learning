import random
import numpy as np

def generateSeeds(n):
    """
    Generates an array of random seeds
    Args:
        n (int) : length of array of random seeds
    Returns:
        rand_arr (np.array(n,)) : array of random seeds
    """
    rand_arr = np.zeros(n,)
    for i in range(n):
        rand_arr[i] = random.randint(-2147483648, 2147483647)
    return rand_arr

def exportSeeds(rand_arr):
    np.savetxt("seeds.csv", rand_arr, delimiter=",")
