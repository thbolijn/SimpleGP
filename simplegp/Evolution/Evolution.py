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
from tqdm import tqdm


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
            max_time=60,
            initialization_max_tree_height=4,
            max_tree_size=20,
            tournament_size=2,
            genetic_algorithm="PSO",
            every_n_generation=1,
            weight_tune_percent=0.5,  # Put to 1 if all trees should be weight-tuned
            weight_tune_selection="random",  # Choose from "best", "worst", "random"
            ga_population_size=40,
            ga_iterations=100,
            de_mutation_rate=0.3,
            de_recombination_rate=0.3
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
        self.de_mutation_rate = de_mutation_rate
        self.de_recombination_rate = de_recombination_rate
        self.ga_iterations = ga_iterations
        self.ga_population_size = ga_population_size

        self.genetic_algorithm = genetic_algorithm
        self.every_n_generation = every_n_generation
        self.weight_tune_percent = weight_tune_percent
        self.weight_tune_selection = weight_tune_selection

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

        # if must_terminate:
        #     print('Terminating at\n\t',
        #           self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t',
        #           np.round(elapsed_time, 2), 'seconds')

        return must_terminate

    def Run(self):
        self.start_time = time.time()

        # List of storing the average tree size of each generation
        avg_tree_size = []

        population = []
        for i in range(self.pop_size):
            population.append(
                Variation.GenerateRandomTree(self.functions, self.terminals, self.initialization_max_tree_height))
            self.fitness_function.Evaluate(population[i])

        repeat = False      # repeat if 3 times in a row the GP gets the same result for elite
        prev_fitness = 0
        count_repeat = 0    # number of times the same fitness value got repeated

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

            # self.show_best_treesize_histogram(selected_population, population)

            if prev_fitness == np.round(self.fitness_function.elite.fitness, 3):
                count_repeat += 1
            else:
                prev_fitness = np.round(self.fitness_function.elite.fitness, 3)
                count_repeat = 0

            if count_repeat >= 2:
                repeat = True

            # if self.generations % self.every_n_generation == 0 and self.generations != 0:
            if repeat and self.genetic_algorithm:

                # Do the selection on the current population
                if self.weight_tune_percent != 1:
                    if self.weight_tune_selection == "random":
                        selection = list(range(len(population)))
                        rnd.shuffle(selection)
                        selected_population = [population[i] for i in
                                               selection[:int(self.weight_tune_percent * len(selection))]]
                    else:
                        fitness_pop = [p.fitness for p in population]
                        arg_fitness = np.argsort(fitness_pop)
                        sorted_population = [population[i] for i in arg_fitness]
                        if self.weight_tune_selection == "best":
                            selection = range(int(len(sorted_population) * self.weight_tune_percent))
                            selected_population = [sorted_population[i] for i in selection]
                        elif self.weight_tune_selection == "worst":
                            total = len(sorted_population)
                            selection = range(int(total - self.weight_tune_percent * total), total)
                            selected_population = [sorted_population[i] for i in selection]
                else:
                    selected_population = population

                # Tune the weights for every tree in the selected population
                print(self.genetic_algorithm, 'tuning on', self.weight_tune_selection, len(selected_population), 'of', len(population), 'trees:')
                for p in selected_population:
                    # if len(p.GetSubtree()) > 1:
                    nodes = p.GetSubtree()
                    W = []  # weight vector
                    for n in nodes:
                        W.append(n.weights)
                    # bounds needed for both algorithms
                    bounds = [(-25, 25)] * len(W) * 2
                    if self.genetic_algorithm == "PSO":
                        pso = PSO(self.fitness_function.Evaluate, W, bounds, p, self.ga_population_size,
                                  self.ga_iterations, self.start_time, self.max_time)
                        W = pso.solution()

                    elif self.genetic_algorithm == "DE":
                        W = DifferentialEvolution.main(self.fitness_function.Evaluate, p, bounds, self.ga_population_size, self.de_mutation_rate,
                                                       self.de_recombination_rate, self.ga_iterations, self.start_time, self.max_time)

                    nodes = p.GetSubtree()
                    for n in nodes:
                        n.weights = [W.pop(), W.pop()]

                    # Break the tuning if the time is up
                    if self.__ShouldTerminate():
                        break

                repeat = False
                count_repeat = 0

            self.generations = self.generations + 1

            sum_sizes = 0
            for p in population:
                sum_sizes += len(p.GetSubtree())
            avg_tree_size.append(sum_sizes / len(population))

            # print('g:', self.generations, 'elite fitness:', np.round(self.fitness_function.elite.fitness, 3), ', size:',
            #       len(self.fitness_function.elite.GetSubtree()))

        # Plot the average tree size over generations
        # plt.plot(avg_tree_size)
        # plt.xlabel('Generation')
        # plt.ylabel('Average tree size')
        # plt.show()

        return self.spreadsheet_string(), population

    def show_treesize_histogram(self, population):
        treesizes = []
        for p in population:
            treesizes.append(len(p.GetSubtree()))
        # print(treesizes)
        plt.hist(treesizes, bins=range(1, self.max_tree_size))
        plt.show()

    def show_best_treesize_histogram(self, best_population, total_population):
        total_treesizes = []
        for p in total_population:
            total_treesizes.append(len(p.GetSubtree()))
        best_treesizes = []
        for p in best_population:
            best_treesizes.append(len(p.GetSubtree()))
        bins = range(1, self.max_tree_size)
        plt.hist(total_treesizes, bins, label="total population")
        plt.hist(best_treesizes, bins, label="best population")
        plt.legend(loc='upper right')
        plt.show()

    def spreadsheet_string(self):
        elapsed_time = np.round(time.time() - self.start_time, 2)
        myList = [self.generations, self.fitness_function.evaluations, elapsed_time, self.pop_size, self.crossover_rate,
                  self.mutation_rate, self.max_evaluations, self.max_generations, self.max_time,
                  self.initialization_max_tree_height, self.max_tree_size, self.tournament_size, self.genetic_algorithm,
                  self.weight_tune_percent, self.weight_tune_selection, self.ga_population_size, self.ga_iterations,
                  self.de_mutation_rate, self.de_recombination_rate]
        result = ','.join(map(str, myList))
        return result

    def show_weight_histogram(self, weights, bounds):
        plt.hist(weights, range=bounds, bins=range(bounds[0], bounds[1], 2))
        plt.show()