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

# def if_then_else(bool, out1, out2):
#     return out1 if bool else out2

pset = gp.PrimitiveSet("MAIN", 5)
# from exec
#pset.addPrimitive(if_then_else, [bool, float, float], float) # I dont think the types are right here

# from int
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

# from bool
#pset.addPrimitive(operator.and_, 2)  # Logical AND
#pset.addPrimitive(operator.or_, 2)   # Logical OR
#pset.addPrimitive(operator.not_, 1)  # Logical NOT

#pset.addPrimitive(operator.greater_or_equal, [float, float], bool)
#pset.addPrimitive(operator.less_or_equal, [float, float], bool)

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

# The the letter grades must be in descending order 
def make_tuple():
    A = random.randint(7, 100)
    B = random.randint(5, A-1)
    C = random.randint(3, B-1)
    D = random.randint(1, C-1)
    G = random.randint(0, 100)
    return (A, B, C, D, G)

def make_inputs(num_inputs, lower_bound, upper_bound):
    return list(make_tuple() for i in range(0, num_inputs))

# Make 100 random input triples where each value v is 0≤v<100.
inputs = make_inputs(100, 0, 100)

def grade(a, b, c, d, g):
    if g < 0: return "Z" # to have better handling of edge cases we could return the value of g
    if g < d: return "F"
    if g < c: return "D"
    if g < b: return "C" 
    if g < a: return "B"
    if g <= 100: return "A"
    if g > 100: return "Z" 

def grading(knownResult, treeResult):
    scale = {
        "A": 5,
        "B": 4,
        "C": 3,
        "D": 2,
        "F": 1,
        }

    if (treeResult == "Z"):
        return 0
    else:
        return 5 - abs((scale[knownResult] - scale[treeResult]))

def evalGrade(individual, points):
    # Compile the individual's tree into a callable function
    func = toolbox.compile(expr=individual)
    
    # Initialize the total error
    errors = 0
    
    # Iterate through all input points
    for A, B, C, D, G in points:
        errors += (grading(
            # actual result, as a letter
            grade(A,B,C,D,G),
            # tree result, as a letter
            grade(A,B,C,D,int(func(A,B,C,D,G))) # func() will return a string (eventually after many changes), but for now its an int
        )**2)
    
    # Return the average error as the fitness value (lower is better)
    return errors / len(points),

    # This computes the average of the squared errors, i.e., the mean squared error,
    # i.e., the MSE.
    return math.fsum(sqerrors) / len(points),

# The training cases are from -4 (inclusive) to +4 (exclusive) in increments of 0.25.
toolbox.register("evaluate", evalGrade, points=inputs)

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

    # Print the best individual
    print("\nBest Individual:")
    best_individual = hof[0]
    print(str(best_individual))

    # Compile the best individual into a callable function
    func = toolbox.compile(expr=best_individual)

        # Print predictions vs actual grades
    print("\nPredictions vs Actual Grades:")
    for A, B, C, D, grade_value in inputs[:8]:  # Limit to 8 examples for readability
        predicted = func(A, B, C, D, grade_value)
        actual = grade(A, B, C, D, grade_value)
        print(f"Inputs: (A={A}, B={B}, C={C}, D={D}, G={grade_value})")
        print(f"Predicted Grade: {grade(A,B,C,D, predicted)}, Actual Grade: {actual}")
        print("-" * 40)

    return pop, log, hof

if __name__ == "__main__":
    main()