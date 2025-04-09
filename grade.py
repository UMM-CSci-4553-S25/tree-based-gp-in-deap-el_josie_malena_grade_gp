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
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.and_, 2)  # Logical AND
pset.addPrimitive(operator.or_, 2)   # Logical OR
pset.addPrimitive(operator.not_, 1)  # Logical NOT

pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))

pset.renameArguments(ARG0='A')
pset.renameArguments(ARG1='B')
pset.renameArguments(ARG2='C')
pset.renameArguments(ARG3='D')
pset.renameArguments(ARG4='G')

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
    if g < 0: return "Yikes!"
    if g < d: return "F"
    if g < c: return "D"
    if g < b: return "C"
    if g < a: return "B"
    return "A"

# Map grades to numeric values for fitness calculation
grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

def evalGrade(individual, points):
    # Compile the individual's tree into a callable function
    func = toolbox.compile(expr=individual)
    
    # Initialize the total error
    errors = 0
    
    # Iterate through all input points
    for A, B, C, D, grade_value in points:
        # Use the individual's function to predict the grade
        predicted = grade(A,B,C,D,func(A, B, C, D, grade_value))
        
        # Get the actual grade using the target function
        actual = grade(A, B, C, D, grade_value)
        
        # Calculate the error as the absolute difference between predicted and actual grades
        # Grades are mapped to numeric values using grade_map
        errors += abs(grade_map.get(predicted, 0) - grade_map[actual])
    
    # Return the average error as the fitness value (lower is better)
    return errors / len(points),

# Register the evaluation function in the toolbox
toolbox.register("evaluate", evalGrade, points=inputs)

# Register the selection operator (tournament selection with size 3)
toolbox.register("select", tools.selTournament, tournsize=3)

# Register the crossover operator (one-point crossover)
toolbox.register("mate", gp.cxOnePoint)

# Register the mutation operator (generate a new subtree using the Full method)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorate the crossover and mutation operators to enforce a maximum tree height of 17
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