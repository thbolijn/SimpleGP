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

np.random.seed(42)

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

# Load regression dataset
df = pd.read_csv(tower, header=None, delimiter=' ')		# Replace the corresponding string for a certain data-set

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.values
y = y.values

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )
# Set fitness function
fitness_function = SymbolicRegressionFitness( X_train, y_train )

# Set functions and terminals
functions = [AddNode(with_weights=True), SubNode(with_weights=True), MulNode(with_weights=True), AnalyticQuotientNode(with_weights=True)]	# chosen function nodes
terminals = [EphemeralRandomConstantNode(with_weights=True)]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

# Run GP
sgp = SimpleGP(fitness_function, functions, terminals, pop_size=100, max_generations=5)	# other parameters are optional
sgp.Run()

# Print results
# Show the evolved function
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.GetSubtree()
print ('Function found (',len(nodes_final_evolved_function),'nodes ):\n\t', nodes_final_evolved_function) # this is in Polish notation
# Print results for training set
print ('Training\n\tMSE:', np.round(final_evolved_function.fitness,3), 
	'\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train),3))
# Re-evaluate the evolved function on the test set
test_prediction = final_evolved_function.GetOutput( X_test )
test_mse = np.mean(np.square( y_test - test_prediction ))
print ('Test:\n\tMSE:', np.round( test_mse, 3), 
	'\n\tRsquared:', np.round(1.0 - test_mse / np.var(y_test),3))
