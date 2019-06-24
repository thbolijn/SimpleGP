import numpy as np
from copy import deepcopy
import time

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train, X_test, y_test, elite_trace_times ):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.elite = None
		self.evaluations = 0
		self.starting_time = time.time()
		self.elite_trace_times = elite_trace_times
		self.elite_trace_train = np.zeros(len(self.elite_trace_times))
		self.elite_trace_test = np.zeros(len(self.elite_trace_times))
		self.elite_counter = 0

	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1

		output = individual.GetOutput( self.X_train )

		mean_squared_error = np.mean ( np.square( self.y_train - output ) )
		individual.fitness = mean_squared_error

		if self.elite_counter < len(self.elite_trace_times) and \
				time.time() - self.starting_time > self.elite_trace_times[self.elite_counter]:
			self.elite_counter += 1
			# print('Incremented counter to ' + str(self.elite_counter) + '. Current MSE: ' + str(self.elite.fitness))

		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)
			# print('new elite with ' + str(mean_squared_error) + ' train MSE')

			# Set training MSE
			self.elite_trace_train[self.elite_counter] = mean_squared_error

			# Set testing MSE
			output = individual.GetOutput(self.X_test)
			mse = np.mean(np.square(self.y_test - output))
			self.elite_trace_test[self.elite_counter] = mse

		return mean_squared_error