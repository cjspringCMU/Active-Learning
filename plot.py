import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Custom modules for simulation and selection strategies
from simulateASR import simulateModels
from selectASR import densitySelect, qbcKNNSelect1, randomSelect, uncertaintyKNNSelect, qbcKNNSelect1Mellow
from seedASR import generateSeeds

def generatePlots(in_file, out_folder, num_simulations):
    # Import data, skipping the header. Assumes data is structured with three columns
    data = np.genfromtxt(in_file, delimiter=',', skip_header=1)
    train = np.empty([0,3])  # Initialize an empty training set
    test = data  # Use the full dataset as the test set initially

    # Generate consistent random seeds for reproducibility across simulation runs
    seeds = generateSeeds(num_simulations)
    knn = KNeighborsClassifier(n_neighbors=5)  # Initialize the KNN classifier with 5 neighbors

    # Simulate using the 'randomSelect' strategy as a baseline
    train_cnts, test_acc_means_random, test_acc_stds_random = simulateModels(num_simulations, data, knn, randomSelect, seeds)
    plt.figure(0)
    plt.errorbar(train_cnts, test_acc_means_random, test_acc_stds_random, marker='o', capsize=2)
    plt.title('Random KNN Test Performance')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.savefig(out_folder + '/random_knn.png')

    # Simulate using the 'uncertaintyKNNSelect' strategy for comparison
    train_cnts, test_acc_means_uncertainty, test_acc_stds_uncertainty = simulateModels(num_simulations, data, knn, uncertaintyKNNSelect, seeds)
    plt.figure(1)
    plt.errorbar(train_cnts, test_acc_means_uncertainty, test_acc_stds_uncertainty, marker='o', capsize=2)
    plt.title('Uncertainty KNN Test Performance')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.savefig(out_folder + '/uncertainty_knn.png')

    # Simulate using the 'densitySelect' strategy to compare density-based selection
    train_cnts, test_acc_means_density, test_acc_stds_density = simulateModels(num_simulations, data, knn, densitySelect, seeds)
    plt.figure(2)
    plt.errorbar(train_cnts, test_acc_means_density, test_acc_stds_density, marker='o', capsize=2)
    plt.title('Density KNN Test Performance')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.savefig(out_folder + '/density_knn.png')

    # Simulate using the 'qbcKNNSelect1' strategy to evaluate Query-by-Committee approaches
    train_cnts, test_acc_means_qbc, test_acc_stds_qbc = simulateModels(num_simulations, data, knn, qbcKNNSelect1, seeds)
    plt.figure(3)
    plt.errorbar(train_cnts, test_acc_means_qbc, test_acc_stds_qbc, marker='o', capsize=2)
    plt.title('QBC KNN Test Performance')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.savefig(out_folder + '/qbc_knn.png')

    # Simulate using a less aggressive version of the QBC strategy ('qbcKNNSelect1Mellow')
    train_cnts, test_acc_means_qbc_mellow, test_acc_stds_qbc_mellow = simulateModels(num_simulations, data, knn, qbcKNNSelect1Mellow, seeds)
    plt.figure(4)
    plt.errorbar(train_cnts, test_acc_means_qbc_mellow, test_acc_stds_qbc_mellow, marker='o', capsize=2)
    plt.title('QBC Mellow KNN Test Performance')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.savefig(out_folder + '/qbc_mellow_knn.png')

    # Compare all aggressive query methods in one plot
    plt.figure(5)
    plt.errorbar(train_cnts, test_acc_means_random, test_acc_stds_random, marker='o', capsize=2, label='random')
    plt.errorbar(train_cnts, test_acc_means_uncertainty, test_acc_stds_uncertainty, marker='o', capsize=2, label="uncertainty")
    plt.errorbar(train_cnts, test_acc_means_density, test_acc_stds_density, marker='o', capsize=2, label="density")
    plt.errorbar(train_cnts, test_acc_means_qbc, test_acc_stds_qbc, marker='o', capsize=2, label="qbc")
    plt.title('KNN Test Performance Across Query Methods')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('knn_agressive.png')

    # Compare mellow and aggressive QBC strategies
    plt.figure(6)
    plt.errorbar(train_cnts, test_acc_means_qbc, test_acc_stds_qbc, marker='o', capsize=2, label="aggressive")
    plt.errorbar(train_cnts, test_acc_means_qbc_mellow, test_acc_stds_qbc_mellow, marker='o', capsize=2, label="mellow")
    plt.title('KNN QBC Test Performance Mellow vs. Aggressive')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(out_folder + '/knn_qbc_mellow_aggressive.png')

import argparse
def main():
    # create the parser
    parser = argparse.ArgumentParser(description="Generate plots from classification data.")
    
    # add the arguments
    parser.add_argument("in_file", type=str, help="The input CSV file containing classification data.")
    parser.add_argument("out_folder", type=str, help="The output folder to save the plots.")
    parser.add_argument("num_simulations", type=int, help="The number of simulations to perform.")
    
    # parse the command line arguments
    args = parser.parse_args()
    
    # call the function with the command-line arguments
    generatePlots(args.in_file, args.out_folder, args.num_simulations)

if __name__ == "__main__":
    main()
