#!/usr/bin/env python3

import random, sys, math, csv
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np
from math import factorial
from deap import creator, base, tools, algorithms
from itertools import chain

# The methode 'simulate' allows to run an algorithm (NSGA-II or NSGA-III) on a benchmark (mOMM, mLOTZ, mOJZJ)
# on individuals with a specified blocksize , a certain rate of individuals to which crossover should be applied (cxpb),
# a population size which equals the size of the Pareto front if not specified otherwise, 
# as well as a maximum number of generations to run the genetic algorithm.
# This method prepares the essential variables required for the DEAP library. 
# The actual execution of the benchmarking is defined by the nsga method called in the last line.

def simulate(algo, M, BLOCK_SIZE, cxpb, factorP=5, POP_SIZE=None, k=2, seed=None, benchmark="OMM", verb=False, NGEN=10000):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,) * M * 2)
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    n = M * BLOCK_SIZE
    
        
    def onesPerBlock(ind): 
        return np.array(ind).reshape((M, BLOCK_SIZE)).sum(axis=1)

    if benchmark == "OMM":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
        def evalFitness(ind):
            ones = onesPerBlock(ind)
            zeros = BLOCK_SIZE - ones
            return np.concatenate((ones, zeros)) #tuple(np.concatenate(ones, zeros))  
        
        def onFront(fitness):
            return True
            
    elif benchmark == "COCZ":
    
        n = 2 * M * BLOCK_SIZE
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
         def evaluate(ind):
            ones = ind.reshape((2*M, BLOCK_SIZE)).sum(axis=1)
            coop = ones[::2]
            comp = ones[1::2]
            onFront = np.all(coop == BLOCK_SIZE)
            return (np.concatenate((coop+comp, coop+BLOCK_SIZE-comp)), onFront) 
        
        def compare(a, b):   # 0 = incomparable, 1 = same objective value, 2 = a strictly dominates b, -1 = b strictly dominates a
            dom_a = np.all(a >= b)
            dom_b = np.all(b >= a)
            
            if dom_a and not dom_b:
                return 2
            if dom_b and not dom_a:
                return -1
            if np.any(a > b):
                return 0
            return 1
        
    elif benchmark == "LOTZ":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
        
        def tz(x):
            return BLOCK_SIZE - numpy.max(numpy.nonzero(numpy.hstack((1, x))))

        def lo(x):
            return numpy.min(numpy.where(numpy.hstack((x, 0)) == 0))
            
        def evalFitness(ind): 
            blocks = ind.reshape((M, BLOCK_SIZE))
            vals = numpy.zeros(2*M, dtype=int)
            for i,block in enumerate(blocks):
                vals[2*i] = lo(block)
                vals[2*i+1] = tz(block)
            return vals
        
        def onFront(fitness):
            return np.all(np.array(fitness).reshape((M, 2)).sum(axis=1) == BLOCK_SIZE)
     
    
    elif benchmark == "OJZJ":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE - 2*k + 3, M)
        def evalFitness(ind):
            ones = onesPerBlock(ind)
            zeros = BLOCK_SIZE - ones
            onejump = ones + k
            onejump[(zeros > 0) & (zeros < k)] = zeros[(zeros > 0) & (zeros < k)]
            zerojump = zeros + k
            zerojump[(ones > 0) & (ones < k)] = ones[(ones > 0) & (ones < k)]
            return np.concatenate((onejump, zerojump),axis=0) #tuple(np.concatenate(onejump, zerojump)) 
        
        def onFront(fitness):
            for o in fitness[:M]:
                if not (o == 0 or o == BLOCK_SIZE or (o >= k and o <= BLOCK_SIZE - k)):
                    return False
            return True
    
    else:
        raise Exception('simulate: The choice of benchmark "{0}" is invalid.'.format(benchmark))
        
        
    MU = POP_SIZE # size of population
    if not MU:
        MU = SIZE_PARETO_FRONT 

    if (algo == "nsga3"):
        P = math.ceil(factorP * n)     
        ref_points = tools.uniform_reference_points(M, P)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalFitness)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/n)
    toolbox.register("mate", tools.cxUniform, indpb=1/2)
     
    if (algo=="nsga3"):
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    elif (algo=="nsga2"):
        toolbox.register("select", my_selNSGA2) 
    else:
         raise Exception('testFor: The choice of algorithm "{0}" is invalid.'.format(algo))
            
    return nsga(toolbox, onFront, BLOCK_SIZE, MU, SIZE_PARETO_FRONT, 0, cxpb, verb=verb, NGEN=NGEN)


# The next block of code contains the adapted implementation of the selection operator of the NSGA-II as described above. 
# The code is exactly the one defined in DEAP/tools except for the highlighted changes.

from operator import attrgetter
from itertools import chain

def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
   # set_trace()
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        random.shuffle(crowd)   # This line is added in comparison to the library, 
                # the following lines are alternated to work on sortedCrowd instead of crowd
        sortedCrowd = sorted(crowd, key=lambda element: element[0][i]) 
        distances[sortedCrowd[0][1]] = float("inf")
        distances[sortedCrowd[-1][1]] = float("inf")
        if sortedCrowd[-1][0][i] == sortedCrowd[0][0][i]:
            continue
        norm = nobj * float(sortedCrowd[-1][0][i] - sortedCrowd[0][0][i])
        for prev, cur, next in zip(sortedCrowd[:-2], sortedCrowd[1:-1], sortedCrowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist
        
def my_selNSGA2(individuals, k, nd='standard'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        random.shuffle(pareto_fronts[-1])  # This line is added in comparison to the library
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen
    
# The next block of code executes the NSGA as defined by its parameters.
# While doing so, it logs and outputs the number of evaluations and the coverage of the Pareto front for each generation.

def coverage(onFront, pop):
    values = [ind.fitness.values for ind in pop if onFront(ind.fitness.values[1:])]
    return len(set(values))

def nsga(toolbox, onFront, BLOCK_SIZE, MU, S, statsTODO, cxpb=0, verb=True, NGEN=1000):

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "cover", "pop"
    
    
    current_cover = 0
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("cover", lambda x: current_cover)
    #stats.register("pop", lambda x: x)
   

    pop = toolbox.population(n=MU)
        
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    current_cover = coverage(onFront, pop)
    cover_history = [current_cover]
    max_cover = current_cover
        
    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if (verb):
        print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN):
        if verb:
            sys.stdout.write('\r' + str(gen) + " " + str(current_cover))
            sys.stdout.flush()
        
        offspring = mutateAndCrossover(pop, toolbox, cxpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        new_pop = toolbox.select(pop + offspring, MU)
        
        
        current_cover = coverage(onFront, pop)
        if current_cover > max_cover:
            max_cover = current_cover
        
        if (gen % S) == 0:
            print("{} - {}".format(gen, gen/10000))
            cover_history.append(current_cover)
        

        pop = new_pop     

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if (verb):
            print(logbook.stream)
        
        if (current_cover == S):
            break
    print()
    return gen, pop, max_cover, cover_history, logbook
    
# The method mutateAndCrossover takes a population and creates an offspring population.
# The population is randomly partioned into pairs of two indivuals and each of these has a chance of cxpb 
# to be replaced by their respective uniform crossover output. 
# Afterward, standard mutation is applied to each individual (no matter whether it is the result of crossover or not).

    
def mutateAndCrossover(population, toolbox, cxpb):
    offspring = [toolbox.clone(ind) for ind in population]
    random.shuffle(offspring)

    if (cxpb > 0):
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
    
    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values
    return offspring
    
    
if __name__== "__main__":
    M = int(sys.argv[1])
    BLOCK_SIZE = int(sys.argv[2])
    cxpb = float(sys.argv[3])
    k = int(sys.argv[4])
    benchmark = sys.argv[5]
    algo = sys.argv[6]
    factorP = float(sys.argv[7])
    psize = int(sys.argv[8])
    POP_SIZE = None if psize <= 0 else psize
    verb = 0 < int(sys.argv[9])
    NGEN = int(sys.argv[10])
    seed = int(sys.argv[11])
    
    
    random.seed(seed)
    np.random.seed(seed)
    
    gen, pop, max_cover, cover_hist, log = simulate(algo, M, BLOCK_SIZE, cxpb, factorP, POP_SIZE, k, benchmark="OMM", verb=verb, NGEN=NGEN)
    print(f"{benchmark},{M},{BLOCK_SIZE},{k},{cxpb},{gen},{NGEN},{seed}")
    