# Active-Learning: Machine Learning Simulation and Plotting Tools

This repository contains a set of Python scripts and modules designed for running machine learning simulations with K-Nearest Neighbors (KNN) classifiers and other tools, using various data selection strategies to enhance model training. It also includes a command-line utility for generating performance plots from classification data.

## Features

- **Data Selection Strategies**: Implements various data selection strategies including Random Selection, Uncertainty Selection, Density-based Selection, and Query by Committee (QBC).
- **KNN Classifier Simulations**: Uses KNN classifiers for simulations with dynamic neighbor selection.
- **Performance Plotting**: Command-line utility to generate and save performance plots based on simulation results.
- **Reproducibility via Seeding**: Uses generated seeds to ensure reproducibility of simulations.

## Components

1. **Data Selection Functions**:
   - `randomSelect`: Moves randomly selected data points from the test set to the training set.
   - `uncertaintyKNNSelect`: Selects data points based on the uncertainty of their classification using a KNN classifier.
   - `densitySelect`: Selects points based on a density measure.
   - `qbcKNNSelect1`, `qbcKNNSelect2`, `qbcKNNSelect1Mellow`: Implements Query by Committee methods using KNN with different settings.
   - `uncertaintyRFSelect`, `qbcRFSelect`: Selection strategies using Random Forest classifiers.

2. **Simulation Functions**:
   - `simulateModels`: Runs multiple simulations to generate mean and standard deviation of performance metrics across different runs.

3. **Plotting Utility**:
   - `main`: A command-line interface that takes file paths and numerical parameters to generate plots using the `generatePlots` function for the KNN methods

## Usage

### Setup
Make sure you have Python installed on your machine along with the necessary libraries (`numpy`, `matplotlib`, `sklearn`). You can install these packages using pip:

```bash
pip install numpy matplotlib scikit-learn
```

### Running Simulations
To run simulations, ensure that your data file (`classification.csv`) is formatted correctly and located in an accessible path. The file is expected to be a csv file with a header and three columns. The first two columns should contain the features while the final column is the label. Then, execute the Python script corresponding to the simulation:

```bash
python simulation_script.py
```

### Generating Plots
To generate plots via the command-line utility, use:

```bash
python plot.py <input_file> <output_directory> <number_of_simulations>
```

Example:
```bash
python plot.py classification.csv ./output 10
```

This will read the classification data, perform the specified number of simulations, and save the resulting plots in the `./output` directory.

## Development

Feel free to fork this repository or submit pull requests with enhancements or fixes. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.
