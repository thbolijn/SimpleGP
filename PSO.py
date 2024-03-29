# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

# from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)

import time

def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc, p):
        nodes = p.GetSubtree()
        positions = deepcopy(self.position_i)
        for n in nodes:
            n.weights = [positions.pop(), positions.pop()]

        self.err_i = costFunc(p)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = deepcopy(self.position_i)
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, x0, bounds, tree_root, num_particles, maxiter, start_time, max_time):

        self.start_time = start_time
        self.max_time = max_time

        nodes = tree_root.GetSubtree()
        W = []  # weight vector
        for n in nodes:
            W.append(n.weights)
            # print('\n Weights: ', n.weights)
        # Here the actual tuning should happen W -> EA -> better W

        x0 = [item for sublist in W for item in sublist]
        # x0 = [random.uniform(-100, 100) for i in range(len(x0))]

        global num_dimensions

        num_dimensions = len(x0)
        self.err_best_g = -1  # best error for group
        self.pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        elapsed_time = 0
        while i < maxiter and elapsed_time < max_time:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc, tree_root)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(swarm[j].position_i)
                    self.err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position(bounds)
            i += 1

            # print('iteration ' + str(i) + ": " + str(self.err_best_g))
            elapsed_time = time.time() - self.start_time
        # print final results
        # print('FINAL:')
        # print(self.pos_best_g)
        # print(self.err_best_g)

    def solution(self):
        return self.pos_best_g


# --- RUN ----------------------------------------------------------------------+

def run():
    initial = [5, 5]  # initial starting location [x1,x2...]
    bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(func1, initial, bounds, num_particles=15, maxiter=50)


# --- PLOT CONTOUR EXAMPLE -----------------------------------------------------+

def plot_contour():
    delta = 0.025
    x = np.arange(-10.0, 10.0, delta)
    y = np.arange(-10.0, 10.0, delta)
    X, Y = np.meshgrid(x, y)
    f = lambda x,y: func1([x,y])
    Z = f(X, Y)

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Simplest default with labels')
    plt.show()


# --- THIS CODE WILL BE EXECUTED -----------------------------------------------+

# run()
# plot_contour()