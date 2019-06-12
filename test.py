# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pandas as pd

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP
import random

seed_no = 42
np.random.seed(seed_no)
random.seed(seed_no)

# Choose the dataset from here
yacht = 'datasets/yacht_full.dat'
white_wine = 'datasets/winewhite_full.dat'
red_wine = 'datasets/winered_full.dat'
airfoil = 'datasets/airfoil_full.dat'
boston = 'datasets/bostonhousing_full.dat'
concrete = 'datasets/concrete_full.dat'
downchemical = 'datasets/downchemical_full.dat'
cooling = 'datasets/energycooling_full.dat'
heating = 'datasets/energyheating_full.dat'
tower = 'datasets/tower_full.dat'

# Parameters
test_size = 0.5
dataset = boston

# Load regression dataset
df = pd.read_csv(dataset, header=None, delimiter=' ')		# Replace the corresponding string for a certain data-set

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.values
y = y.values



train_mse_list = []
test_mse_list = []
train_R_list = []
test_R_list = []

no_repetitions = 2

for whatever in range(no_repetitions):

    # Take a dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_no)
    # Set fitness function
    fitness_function = SymbolicRegressionFitness(X_train, y_train)

    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node
    for i in range(X.shape[1]):
        terminals.append(FeatureNode(i))  # add a feature node for each feature

    # Run GP
    sgp = SimpleGP(fitness_function, functions, terminals)  # other parameters are optional
    spreadsheet_string = sgp.Run()

    # Print results
    # Show the evolved function
    final_evolved_function = fitness_function.elite
    nodes_final_evolved_function = final_evolved_function.GetSubtree()
    print('Function found (', len(nodes_final_evolved_function), 'nodes ):\n\t', nodes_final_evolved_function)  # this is in Polish notation
    # Print results for training set
    training_MSE = np.round(final_evolved_function.fitness, 3)
    training_Rsquared = np.round(1.0 - final_evolved_function.fitness / np.var(y_train), 3)
    print('Training\n\tMSE:', training_MSE,
        '\n\tRsquared:', training_Rsquared)
    # Re-evaluate the evolved function on the test set
    test_prediction = final_evolved_function.GetOutput(X_test)
    test_mse = np.round(np.mean(np.square(y_test - test_prediction)), 3)
    test_Rsquared = np.round(1.0 - test_mse / np.var(y_test), 3)
    print('Test:\n\tMSE:', test_mse,
        '\n\tRsquared:', test_Rsquared)

    train_mse_list.append(training_MSE)
    test_mse_list.append(test_mse)
    train_R_list.append(training_Rsquared)
    test_R_list.append(test_Rsquared)


avg_train_mse = np.mean(train_mse_list)
std_train_mse = np.std(train_mse_list)

avg_test_mse = np.mean(test_mse_list)
std_test_mse = np.std(test_mse_list)

avg_train_R = np.mean(train_R_list)
std_train_R = np.std(train_R_list)

avg_test_R = np.mean(test_R_list)
std_test_R = np.std(test_R_list)

print(f'Train mean MSE: {avg_train_mse}\t Train std MSE: {std_train_mse}'
      f'\nTrain mean R squared: {avg_train_R}\t Train std R squared: {std_train_R}'
      f'\nTest mean MSE: {avg_test_mse}\t Test std MSE: {std_test_mse}'
      f'\nTest mean R squared: {avg_test_R}\t Test std R squared: {std_test_R}')

result = ','.join(map(str, [dataset, seed_no, test_size, np.round(avg_train_mse, 3), np.round(avg_train_R, 3),
							np.round(avg_test_mse, 3), np.round(avg_test_R, 3), no_repetitions, spreadsheet_string]))
print(result)