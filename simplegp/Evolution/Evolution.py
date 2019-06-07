import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy
import random as rnd

from simplegp.Variation import Variation
from simplegp.Selection import Selection
from simplegp.DifferentialEvolution import DifferentialEvolution
from PSO import PSO
import matplotlib.pyplot as plt


class SimpleGP:

    def __init__(
            self,
            fitness_function,
            functions,
            terminals,
            pop_size=500,
            crossover_rate=0.5,
            mutation_rate=0.5,
            max_evaluations=-1,
            max_generations=-1,
            max_time=-1,
            initialization_max_tree_height=10,
            max_tree_size=100,
            tournament_size=4,
            genetic_algorithm="DE",
            every_n_generation=1,
            best_top_percent=0.1
    ):

        self.pop_size = pop_size
        self.fitness_function = fitness_function
        self.functions = functions
        self.terminals = terminals
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.max_time = max_time
        self.initialization_max_tree_height = initialization_max_tree_height
        self.max_tree_size = max_tree_size
        self.tournament_size = tournament_size

        self.genetic_algorithm = genetic_algorithm
        self.every_n_generation = every_n_generation
        self.best_top_percent = best_top_percent

        self.generations = 0


    def __ShouldTerminate(self):
        must_terminate = False
        elapsed_time = time.time() - self.start_time
        if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
            must_terminate = True
        elif self.max_generations > 0 and self.generations >= self.max_generations:
            must_terminate = True
        elif self.max_time > 0 and elapsed_time >= self.max_time:
            must_terminate = True
        if must_terminate:
            print('Terminating at\n\t',
                  self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')
        return must_terminate

    def Run(self):

        self.start_time = time.time()

        population = []
        for i in range(self.pop_size):
            population.append(
                Variation.GenerateRandomTree(self.functions, self.terminals, self.initialization_max_tree_height))
            self.fitness_function.Evaluate(population[i])

        while not self.__ShouldTerminate():

            O = []

            for i in range(len(population) - 1):

                o = deepcopy(population[i])
                if (random() < self.crossover_rate):
                    o = Variation.SubtreeCrossover(o, population[randint(len(population))])
                if (random() < self.mutation_rate):
                    o = Variation.SubtreeMutation(o, self.functions, self.terminals,
                                                  max_height=self.initialization_max_tree_height)

                if len(o.GetSubtree()) > self.max_tree_size:
                    del o
                    o = deepcopy(population[i])
                else:
                    self.fitness_function.Evaluate(o)

                O.append(o)

            PO = population + O
            population = Selection.TournamentSelect(PO, len(population), tournament_size=self.tournament_size)

            fitness_pop = [p.fitness for p in population]
            arg_fitness = np.argsort(fitness_pop)
            sorted_population = [population[i] for i in arg_fitness]
            best_sorted = [sorted_population[i] for i in range(int(len(sorted_population) * self.best_top_percent))]

            self.show_treesize_histogram(population)

            if self.generations % self.every_n_generation == 0 and self.generations != 0:
                for p in best_sorted:
                    if len(p.GetSubtree()) > 1:

                        nodes = p.GetSubtree()
                        W = []  # weight vector
                        for n in nodes:
                            W.append(n.weights)
                        # 	print('\n Weights: ', n.weights)
                        bounds = [(-10, 10)] * len(W) * 2

                        if self.genetic_algorithm == "PSO":
                            num_particles = 40
                            max_iterations = 100
                            pso = PSO(self.fitness_function.Evaluate, W, bounds, p, num_particles, max_iterations)
                            W = pso.solution()

                        if self.genetic_algorithm == "DE":
                            popsize = 40
                            mutate = 0.5
                            recombination = 0.3
                            maxiter = 100
                            W = DifferentialEvolution.main(self.fitness_function.Evaluate, p, bounds, popsize, mutate,
                                                           recombination, maxiter)

                        self.show_weight_histogram(W, bounds[0])

                        nodes = p.GetSubtree()
                        for n in nodes:
                            n.weights = [W.pop(), W.pop()]


            self.generations = self.generations + 1
            # Here the weights tuning should happen

            self.generations = self.generations + 1

            print('g:', self.generations, 'elite fitness:', np.round(self.fitness_function.elite.fitness, 3), ', size:',
                  len(self.fitness_function.elite.GetSubtree()))

    def show_treesize_histogram(self, population):
        treesizes = []
        for p in population:
            treesizes.append(len(p.GetSubtree()))
        # print(treesizes)
        plt.hist(treesizes, bins=range(1, self.max_tree_size))
        plt.show()

    def show_weight_histogram(self, weights, bounds):
        plt.hist(weights, range=bounds, bins=range(bounds[0], bounds[1], 2))
        plt.show()