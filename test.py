# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pandas as pd
import time
import os
from Plot import experiment, plot_experiment

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP
import random
from tqdm import tqdm

# seed_no = 42
# np.random.seed(seed_no)
# random.seed(seed_no)

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
train_trace_list = []
test_trace_list = []

no_repetitions = 5

pop_size = 2048
crossover_rate = 0.75
mutation_rate = 1
max_tree_size = 20
tournament_size = 4
ga_pop_size = 50
genetic_algorithm_list = ['PSO', 'DE', None]
de_recombination_rate = 0.25
de_mutation_rate = 0.75
max_time = 600

seeds = [42, 166, 214, 5, 82, 13, 22, 109, 96, 1888, 12345, 667, 68, 1995, 988, 4321, 421, 8888, 500, 10]

trace_timing = range(1, 601)

# Initialize weight tuning parameters
weight_tune_percent = 0
weight_tune_selection = ''

for genetic_algorithm in genetic_algorithm_list:
    for index in tqdm(range(len(seeds))):

        np.random.seed(seeds[index])
        random.seed(seeds[index])

        if genetic_algorithm == 'DE':
            weight_tune_percent = 0.05
            weight_tune_selection = 'random'
        elif genetic_algorithm == 'PSO':
            weight_tune_percent = 0.5
            weight_tune_selection = 'random'
        else:
            weight_tune_percent = 0
            weight_tune_selection = 'random'


        # Take a dataset split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seeds[index])
        # Set fitness function
        fitness_function = SymbolicRegressionFitness(X_train, y_train, X_test, y_test, trace_timing)

        # Set functions and terminals
        functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
        terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node
        for i in range(X.shape[1]):
            terminals.append(FeatureNode(i))  # add a feature node for each feature

        # Run GP
        sgp = SimpleGP(fitness_function, functions, terminals,
                       pop_size=pop_size, crossover_rate=crossover_rate,
                       mutation_rate=mutation_rate, max_tree_size=max_tree_size,
                       tournament_size=tournament_size, de_recombination_rate=de_recombination_rate,
                       de_mutation_rate=de_mutation_rate, max_time=max_time,
                       weight_tune_percent=weight_tune_percent, weight_tune_selection=weight_tune_selection,
                       ga_population_size=ga_pop_size, genetic_algorithm=genetic_algorithm)
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

        [train_trace, test_trace] = fitness_function.get_processed_traces()
        train_trace_list.append(train_trace)
        test_trace_list.append(test_trace)

    avg_train_mse = np.mean(train_mse_list)
    std_train_mse = np.std(train_mse_list)

    avg_test_mse = np.mean(test_mse_list)
    std_test_mse = np.std(test_mse_list)

    avg_train_R = np.mean(train_R_list)
    std_train_R = np.std(train_R_list)

    avg_test_R = np.mean(test_R_list)
    std_test_R = np.std(test_R_list)

    # Saving MSE for plotting
    train_trace_mean = np.mean(train_trace_list, axis=0)
    test_trace_mean = np.mean(test_trace_list, axis=0)
    train_trace_std = np.std(train_trace_list, axis=0)
    test_trace_std = np.std(test_trace_list, axis=0)
    if not os.path.exists('results'):
        os.mkdir('results')
    time_string = time.strftime('%Y%m%d_%H%M%S_', time.localtime())
    filename = 'results/' + time_string + 'traces_' + genetic_algorithm
    traces = np.vstack([trace_timing, train_trace_mean, train_trace_std, test_trace_mean, test_trace_std])
    np.save(filename, traces)

    # Plotting training and testing MSE
    e = experiment(filename + '.npy')
    plot_experiment([e])

    print(f'Train mean MSE: {avg_train_mse}\t Train std MSE: {std_train_mse}'
          f'\nTrain mean R squared: {avg_train_R}\t Train std R squared: {std_train_R}'
          f'\nTest mean MSE: {avg_test_mse}\t Test std MSE: {std_test_mse}'
          f'\nTest mean R squared: {avg_test_R}\t Test std R squared: {std_test_R}')

    result = ','.join(map(str, [dataset, test_size, np.round(avg_train_mse, 3), np.round(avg_train_R, 3),
                                np.round(avg_test_mse, 3), np.round(avg_test_R, 3), no_repetitions, spreadsheet_string]))
    print(result)