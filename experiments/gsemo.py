#!/usr/bin/env python3

import random, sys, math, csv
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np

def simulate(M, BLOCK_SIZE, cxpb, k=2, benchmark="OMM", verb=False, NGEN=10000):
    
    n = M * BLOCK_SIZE
    
    def onesPerBlock(ind): 
        return ind.reshape((M, BLOCK_SIZE)).sum(axis=1)

    if benchmark == "OMM":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
        def evaluate(ind):  # only tracks number of 1-bits per block, fitness is implicitly evaluated in compare(a,b)
            return (onesPerBlock(ind), True)
        
        def compare(a, b):   # 0 = incomparable, 1 = same objective value, 2 = a strictly dominates b, -1 = b strictly dominates a
            if np.all(a == b):
                return 1
            else:
                return 0
                
    elif benchmark == "COCZ":
    
        n = 2 * M * BLOCK_SIZE
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
         def evaluate(ind):
            ones = ind.reshape((2*M, BLOCK_SIZE)).sum(axis=1)
            coop = ones[::2]
            comp = ones[1::2]
            onFront = np.all(coop == BLOCK_SIZE)
            return (np.concatenate((coop+comp, coop+BLOCK_SIZE-comp)), onFront) # invert for minimization 
        
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
            return BLOCK_SIZE - np.max(np.nonzero(np.hstack((1, x))))

        def lo(x):
            return np.min(np.where(np.hstack((x, 0)) == 0))

        def evaluate(ind):  # tuple. first element is array of fitness values individual objectives. Second element stores whether it is on the Pareto front 
            blocks = ind.reshape((M, BLOCK_SIZE))
            onFront = True
            vals = np.zeros(2*M, dtype=int)
            for i,block in enumerate(blocks):
                vals[2*i] = lo(block)
                vals[2*i+1] = tz(block)
                if vals[2*i] + vals[2*i+1] < BLOCK_SIZE:
                    onFront = False
            return (vals, onFront)
        
        def compare(a, b):
            dom_a = np.all(a >= b)
            dom_b = np.all(b >= a)
            
            if dom_a and not dom_b:
                return 2
            if dom_b and not dom_a:
                return -1
            if np.any(a > b):
                return 0
            return 1
    
    elif benchmark == "OJZJ":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE - 2*k + 3, M)
        
        def evaluate(ind):   # only tracks number of 1-bits per block, fitness is implicitly evaluated in compare(a,b)
            ones = onesPerBlock(ind)
            onFront = True
            for o in ones:
                if not (o == 0 or o == BLOCK_SIZE or (o >= k and o <= BLOCK_SIZE - k)):
                    onFront = False
            return (ones, onFront)
            
        def compare(a, b): 
            str_dom_a = np.any(((a > b) & (b < k) & (b > 0)) | ((a < b) & (b > BLOCK_SIZE-k) & (b < BLOCK_SIZE)))
            str_dom_b = np.any(((b > a) & (a < k) & (a > 0)) | ((b < a) & (a > BLOCK_SIZE-k) & (a < BLOCK_SIZE)))
            
            if str_dom_a and not str_dom_b:
                return 2
            if str_dom_b and not str_dom_a:
                return -1
            if np.all(a == b):
                return 1
            return 0 
    else:
        raise Exception('simulate: The choice of benchmark "{0}" is invalid.'.format(benchmark))
            
    return gsemo(compare, evaluate, BLOCK_SIZE, M, n, SIZE_PARETO_FRONT, NGEN, cxpb, verb)
    
    

def mutate(ind, flip_chance):
    flip_mask = np.array(np.random.choice([0,1], p=[1-flip_chance, flip_chance], size = ind.size), dtype=bool)
    return np.logical_xor(ind, flip_mask, out=ind.copy())

def crossover(ind1, ind2):
    co_mask = np.array(np.random.choice([0,1], p=[0.5, 0.5], size = ind1.shape), dtype=bool)
    ind1_tmp = np.logical_and(ind1, co_mask, out=ind1.copy())
    ind2_tmp = np.logical_and(ind2, np.invert(co_mask), out=ind2.copy())
    return np.logical_or(ind1_tmp, ind2_tmp, out=ind1_tmp)

def mutateAndCrossover(pop, flip_chance, cxpb):
    if (len(pop) >= 2 and random.random() < cxpb):
        parents = random.sample(pop, 2)
        parent = crossover(parents[0][0], parents[1][0])
    else: 
        parent = random.sample(pop, 1)[0][0]
    return mutate(parent, flip_chance)
    
    
def gsemo(compare, evaluate, BLOCK_SIZE, M, n, SIZE_PARETO_FRONT, NGEN, cxpb, verb):
        
    if verb:
        print("Running GSEMO")    
        
    # pop_cover[x_1]...[x_M] is the number of individuals in the population with exactly x_i many 1-bits in the ith block for all i 
    pop_cover = np.zeros((BLOCK_SIZE+1,)*M, dtype=int) 

    init_string = np.random.randint(2, size=n)
    obj_value = evaluate(init_string)
    pop = [(init_string,obj_value)]
    cover_count = 1 if obj_value[1] else 0    
    cover_history = [cover_count]

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = mutateAndCrossover(pop, 1/n, cxpb)
        obj_value = evaluate(offspring)
            
        dominated = False
        new_value = True
        for i, ind in enumerate(pop):
            comparison = compare(obj_value[0], ind[1][0])
            if comparison >= 1:   # This is the variant where for the same objective value the newest individual is kept
                pop.pop(i)
                
                if comparison == 1:
                    new_value = False
            elif comparison == -1:
                dominated = True
                break
        if not dominated:
           pop.append((offspring,obj_value))
            if new_value and obj_value[1]:
                cover_count += 1
        # Compile statistics about the new population
        
        if (gen % SIZE_PARETO_FRONT == 0):
            cover_history.append(cover_count)
            if (verb):
                print("Generation {} of size {} covers {} out of {}".format(gen, len(pop), cover_count, SIZE_PARETO_FRONT))
        
        if (cover_count == SIZE_PARETO_FRONT):
            break
    if (verb):
        print("Done after {} iterations\n".format(gen))
    return gen, pop
    
   
 
if __name__== "__main__":
    M = int(sys.argv[1])
    BLOCK_SIZE = int(sys.argv[2])
    cxpb = float(sys.argv[3])
    k = int(sys.argv[4])
    benchmark = sys.argv[5]
    verb = 0 < int(sys.argv[6])
    NGEN = int(sys.argv[7])
    seed = int(sys.argv[8])
    
    random.seed(seed)
    np.random.seed(seed)
    
    gen, pop = simulate(M, BLOCK_SIZE, cxpb, k, benchmark=benchmark, verb=verb, NGEN=NGEN)
    print(f"{benchmark},{M},{BLOCK_SIZE},{k},{cxpb},{gen},{NGEN},{seed}")
    