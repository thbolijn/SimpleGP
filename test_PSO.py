# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import random
from copy import deepcopy
import pandas as pd
from tqdm import tqdm

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP

seeds = [42, 166, 214, 5, 82]
pop_size = 2048
crossover_rate = 0.75
mutation_rate = 1
max_tree_size = 20
tournament_size = 4
genetic_algorithm = 'PSO'

ga_pop_size_list = [50, 500]
de_mutation_rate_list = [0.25, 0.5, 0.75]
de_crossover_rate_list = [0.25, 0.5, 0.75]
weight_tune_percent_list = [0.05, 0.1, 0.25, 0.5, 0.75]
weight_tune_selection_list = ['best', 'random']



no_runs = 5

results = []  	# List of strings with the results from each test

for ga_pop_size in ga_pop_size_list:
    for weight_tune_percent in weight_tune_percent_list:
        for weight_tune_selection in weight_tune_selection_list:

            # No matter the selection criterion, if we take 100% of trees the result will be the same
            if (weight_tune_selection != 'best' and ga_pop_size == 500) or (weight_tune_selection != 'random' and ga_pop_size == 50):
                break

            train_mse_list = []
            test_mse_list = []
            train_Rsquared_list = []
            test_Rsquared_list = []

            no_nodes_list = []

            avg_train_mse = 0
            std_train_mse = 0
            avg_train_Rsquared = 0
            std_train_Rsquared = 0

            avg_test_mse = 0
            std_test_mse = 0
            avg_test_Rsquared = 0
            std_test_Rsquared = 0

            avg_no_nodes = 0

            for run in tqdm(range(no_runs)):

                np.random.seed(seeds[run])
                random.seed(seeds[run])
                # Load regression dataset
                X, y = sklearn.datasets.load_boston(return_X_y=True)
                # Take a dataset split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seeds[run])
                # Set fitness function
                fitness_function = SymbolicRegressionFitness( X_train, y_train )

                # Set functions and terminals
                functions = [ AddNode(), SubNode(), MulNode(), AnalyticQuotientNode() ]	# chosen function nodes
                terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
                for i in range(X.shape[1]):
                    terminals.append(FeatureNode(i))	# add a feature node for each feature

                # Run GP
                sgp = SimpleGP(fitness_function, functions, terminals,
                               pop_size=pop_size, crossover_rate=crossover_rate,
                               mutation_rate=mutation_rate, max_tree_size=max_tree_size,
                               tournament_size=tournament_size, max_time=600,
                               ga_population_size=ga_pop_size, genetic_algorithm=genetic_algorithm,
                               weight_tune_percent=weight_tune_percent, weight_tune_selection=weight_tune_selection)  # other parameters are optional
                sgp.Run()

                # Print results
                # Show the evolved function
                final_evolved_function = fitness_function.elite
                nodes_final_evolved_function = final_evolved_function.GetSubtree()

                test_prediction = final_evolved_function.GetOutput( X_test )
                test_mse = np.mean(np.square( y_test - test_prediction ))

                train_mse_list.append(np.round(final_evolved_function.fitness, 3))
                train_Rsquared_list.append(np.round(1.0 - final_evolved_function.fitness / np.var(y_train), 3))
                test_mse_list.append(test_mse)
                test_Rsquared_list.append(np.round(1.0 - test_mse / np.var(y_test), 3))

                no_nodes_list.append(len(nodes_final_evolved_function))

            avg_train_mse = np.mean(train_mse_list)
            std_train_mse = np.std(train_mse_list)
            avg_train_Rsquared = np.mean(train_Rsquared_list)
            std_train_Rsquared = np.std(train_Rsquared_list)

            avg_test_mse = np.mean(test_mse_list)
            std_test_mse = np.std(test_mse_list)
            avg_test_Rsquared = np.mean(test_Rsquared_list)
            std_test_Rsquared = np.std(test_Rsquared_list)

            avg_no_nodes = np.mean(no_nodes_list)

            results.append(f'{pop_size}, {crossover_rate}, {mutation_rate}, {max_tree_size}, {tournament_size},'
                           f' {genetic_algorithm}, {ga_pop_size}, {weight_tune_percent}, {weight_tune_selection}'
                           f' {avg_no_nodes}, {avg_train_mse}, {std_train_mse}, {avg_train_Rsquared},'
                           f' {std_train_Rsquared}, {avg_test_mse}, {std_test_mse}, {avg_test_Rsquared},'
                           f' {std_test_Rsquared}')


df = pd.DataFrame(results, columns=['pop_size, crossover_rate, mutation_rate, max_tree_size, tournament_size,'
                                    'genetic_algorithm, ga_pop_size, weight_tune_percent, weight_tune_selection,'
                                    ' no_nodes,'
									'train_mse_avg, train_mse_std, train_Rsquared_avg, train_Rsquared_std,'
									'test_mse_avg, test_mse_std, test_Rsquared_avg, test_Rsquared_std'])
df.to_csv('results_PSO.csv', index=False)
