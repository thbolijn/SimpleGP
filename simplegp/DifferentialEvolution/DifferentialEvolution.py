# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
from copy import deepcopy
import time

# --- EXAMPLE COST FUNCTIONS ---------------------------------------------------+

def func1(x):
    # Sphere function, use any bounds, f(0,...,0)=0
    return sum([x[i] ** 2 for i in range(len(x))])


def func2(x):
    # Beale's function, use bounds=[(-4.5, 4.5),(-4.5, 4.5)], f(3,0.5)=0.
    term1 = (1.500 - x[0] + x[0] * x[1]) ** 2
    term2 = (2.250 - x[0] + x[0] * x[1] ** 2) ** 2
    term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    return term1 + term2 + term3


# --- FUNCTIONS ----------------------------------------------------------------+


def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])

    return vec_new


# --- MAIN ---------------------------------------------------------------------+

def main(cost_func, p, bounds, popsize, mutate, recombination, maxiter, start_time, max_time):

    # --- INITIALIZE A POPULATION (step #1) ----------------+

    population = []
    newPopulation = [] #generation+1
    for i in range(0, popsize):
       indv = []
       for j in range(len(bounds)):
           indv.append(random.uniform(bounds[j][0], bounds[j][1]))
       population.append(indv)

    # --- SOLVE --------------------------------------------+

    # nodes = p.GetSubtree()
    # W = []  # weight vector
    # for n in nodes:
    #     W.append(n.weights)
    #     print('\n Weights: ', n.weights)
    # cycle through each generation (step #2)
    elapsed_time = 0
    for i in range(1, maxiter + 1):
        #print('GENERATION:', i)

        gen_scores = []  # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):

            # --- MUTATION (step #3.A) ---------------------+

            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0, popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]  # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            # --- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])

            # --- GREEDY SELECTION (step #3.C) -------------+
            nodes = p.GetSubtree()

            positions = deepcopy(v_trial)
            for n in nodes:
                n.weights = [positions.pop(), positions.pop()]
            score_trial = cost_func(p)

            positions = deepcopy(x_t) # TODO: Is double deepcopy needed or can positions just be updated? Does it matter?
            # positions = x_t
            for n in nodes:
                n.weights = [positions.pop(), positions.pop()]

            score_target = cost_func(p)

            if score_trial < score_target:
                newPopulation.append(v_trial)
                gen_scores.append(score_trial)
                #print('   >', score_trial, v_trial)

            else:
                newPopulation.append(x_t)
                #print('   >', score_target, x_t)
                gen_scores.append(score_target)

            elapsed_time = time.time() - start_time

            if elapsed_time > max_time:
                break

        # --- SCORE KEEPING --------------------------------+
        population = newPopulation
        #gen_avg = sum(gen_scores) / popsize  # current generation avg. fitness
        gen_best = min(gen_scores)  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]  # solution of best individual

        if elapsed_time > max_time:
            break

    return gen_sol


# --- CONSTANTS ----------------------------------------------------------------+

cost_func = func1  # Cost function
bounds = [(-1, 1), (-1, 1)]  # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
popsize = 10  # Population size, must be >= 4
mutate = 0.5  # Mutation factor [0,2]
recombination = 0.7  # Recombination rate [0,1]
maxiter = 20  # Max number of generations (maxiter)

# --- RUN ----------------------------------------------------------------------+

#x = main(cost_func, bounds, popsize, mutate, recombination, maxiter)

# --- END ----------------------------------------------------------------------+