#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

# This is an implementation of the "median" problem from Tom Helmuth's
# software synthesis benchmark suite (PSB1):
#
# T. Helmuth and L. Spector. General Program Synthesis Benchmark Suite. In
# GECCO '15: Proceedings of the 17th annual conference on Genetic and
# evolutionary computation. July 2015. ACM.
#
# Problem Source: C. Le Goues et al., "The ManyBugs and IntroClass
# Benchmarks for Automated Repair of C Programs," in IEEE Transactions on
# Software Engineering, vol. 41, no. 12, pp. 1236-1256, Dec. 1 2015.
# doi: 10.1109/TSE.2015.2454513
#
# The goal of this problem is to take three integer inputs and return the
# median (middle) of those three values.
#
# This problem is quite easy if you have both `Min` and `Max` instructions,
# or a `Clamp` instruction, but can be more difficult without those
# instruction.

import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def square(x):
    return x*x

def double(x):
    return x+x

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))

pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
pset.renameArguments(ARG3='d')
pset.renameArguments(ARG4='g')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# `genHalfAndHalf` generates half the initial population using the `Full` method and half
# using the `Grow` method. `Full` picks a random depth (between min and max) and generates
# a _full_ tree of that depth. `Grow` picks a random depth, and grows a tree until, along
# each branch, either a leaf is chosen or the maximum depth is reached.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def make_tuple():
    A = random.randint(4, 101)
    B = random.randint(3, A)
    C = random.randint(2, B)
    D = random.randint(1, C)
    G = random.randint(0, 101)
    return (A, B, C, D, G)

def make_inputs(num_inputs, lower_bound, upper_bound):
    return list(make_tuple() for i in range(0, num_inputs))

# Make 100 random input triples where each value v is 0≤v<100.
inputs = make_inputs(100, 0, 100)

def grade(a, b, c, d, g):
    values = [a, b, c, d, g]
    #values.sort()
    return values[4]

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x^9 + 3x^6 + 3x^3 + 2
    # `sqerrors` is the square of all the errors.
    sqerrors = ((func(a, b, c, d, e) - median(a, b, c, d, e))**2 for (a, b, c, d, e) in points)

    # This computes the average of the squared errors, i.e., the mean squared error,
    # i.e., the MSE.
    return math.fsum(sqerrors) / len(points),

# The training cases are from -4 (inclusive) to +4 (exclusive) in increments of 0.25.
toolbox.register("evaluate", evalSymbReg, points=inputs)

# Tournament selection with tournament size 3
toolbox.register("select", tools.selTournament, tournsize=3)

# One-point crossover, i.e., remove a randomly chosen subtree from the parent,
# and replace it with a randomly chosen subtree from a second parent.
toolbox.register("mate", gp.cxOnePoint)

# Remove a randomly chosen subtree and replace it with a "full" randomly generated
# tree whose depth is between min and max. `expr_mut` is specifying a way of
# generating new trees using `Full`. `mutUniform` below says that `mutate` will
# _uniformly_ choose a subtree to remove and replace it using `expr_mut`, i.e.,
# `Full`.
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Set the maximum height of a tree after either crossover or mutation to be 17.
# When an invalid (over the limit) child is generated, it is simply replaced
# by one of its parents, randomly selected. This replacement policy is a Real Bad Idea
# because it rewards parents who are likely to create offspring that are above the
# threshold.
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    # random.seed(318)

    # Sets the population size to 300.
    pop = toolbox.population(n=300)
    # Tracks the single best individual over the entire run.
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Does the run, going for 40 generations (the 5th argument to `eaSimple`).
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 200, stats=mstats,
                                   halloffame=hof, verbose=True)

    # print log

    # Print the members of the hall of fame
    for winner in hof:
        print(str(winner))

    return pop, log, hof

if __name__ == "__main__":
    main()