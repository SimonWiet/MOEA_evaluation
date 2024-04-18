#!/usr/bin/env python3

import random, sys, math, csv
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np
import pygmo as pg
from pygmo import hypervolume

def simulate(M, BLOCK_SIZE, cxpb, k=2, benchmark="OMM", pop_factor=1, verb=False, NGEN=10000):
    
    n = M * BLOCK_SIZE
    
    def onesPerBlock(ind): 
        return ind.reshape((M, BLOCK_SIZE)).sum(axis=1)
        
        
    #Formulated for minimization
    

    if benchmark == "OMM":
    
        max_val = BLOCK_SIZE
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
        def evaluate(ind): 
            ones = onesPerBlock(ind)
            return (np.concatenate((ones, BLOCK_SIZE - ones)), True)
            
            
    elif benchmark == "COCZ":
        
        max_val = BLOCK_SIZE * 2
    
        n = 2 * M * BLOCK_SIZE
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, M)
        
        def evaluate(ind):
            coop = ind[:BLOCK_SIZE*M].sum()
            ones = ind[BLOCK_SIZE*M:].reshape((M, BLOCK_SIZE)).sum(axis=1)
            onFront = (coop == M*BLOCK_SIZE)
            return ((M+1)*BLOCK_SIZE - coop - np.concatenate((ones, BLOCK_SIZE-ones)), onFront) # invert for minimization 


      
        
    elif benchmark == "LOTZ":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE + 1, 2*M-1)
        
        max_val = BLOCK_SIZE
        
        def tz(x):
            return BLOCK_SIZE - np.max(np.nonzero(np.hstack((1, x))))

        def lo(x):
            return np.min(np.where(np.hstack((x, 0)) == 0))

        def evaluate(ind): 
            blocks = ind.reshape((M, BLOCK_SIZE))
            vals = np.zeros(2*M, dtype=int)
            onFront = True
            for i,block in enumerate(blocks):
                vals[2*i] = lo(block)
                vals[2*i+1] = tz(block)
                if vals[2*i] + vals[2*i+1] < BLOCK_SIZE:
                    onFront = False
            return (BLOCK_SIZE - vals, onFront) # invert for minimization
        
    
    elif benchmark == "OJZJ":
        
        SIZE_PARETO_FRONT = pow(BLOCK_SIZE - 2*k + 3, M)
        
        max_val = BLOCK_SIZE + k
        
        def evaluate(ind):   
            ones = ind.reshape((M, BLOCK_SIZE)).sum(axis=1)
            vals = np.zeros(2*M, dtype=int)
            onFront = True
            
            for i,o in enumerate(ones):

                if o <= BLOCK_SIZE - k or o == BLOCK_SIZE: 
                    vals[2*i] = o + k
                else: 
                    vals[2*i] = BLOCK_SIZE - o
                    onFront = False

                if o >= k or o == 0: 
                    vals[2*i+1] = BLOCK_SIZE - o + k
                else: 
                    vals[2*i+1] = o
                    onFront = False
            return (BLOCK_SIZE + k - vals, onFront)    #invert for minimization
            
    else:
        raise Exception(f'simulate: The choice of benchmark "{benchmark}" is invalid.')
        
    if pop_factor < 1:
        raise Exception(f'simulate: pop_factor has to be at least 1, but is {pop_factor}')
        
            
    return smsemoa(evaluate, BLOCK_SIZE, max_val, M, n, pop_factor, SIZE_PARETO_FRONT, NGEN, cxpb, verb)
    
    

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
    
    
def smsemoa(evaluate, BLOCK_SIZE, max_val, M, n, pop_factor, SIZE_PARETO_FRONT, NGEN, cxpb, verb):
        
    if verb:
        print("Running SMS-EMOA")    
      
    pop_size = int(SIZE_PARETO_FRONT * pop_factor)
    pop = []
    for i in range(pop_size):
        init_string = np.random.randint(2, size=n)
        evalue = evaluate(init_string)
        pop.append((init_string,evalue))
    
    # The set of found elements on the Pareto front. As we only assured pop_size >= Pareto front, no value will be lost from this set
    represented = set([tuple(val[0]) for x,val in pop if val[1]])  
  
    
    cover_count = len(represented)
    cover_history = [cover_count]
   
    ref_point = np.full((2*M,), max_val+1)  # for hypervolume contribution
    
    # Begin the generational process
    
    
    for gen in range(1, NGEN):
        offspring = mutateAndCrossover(pop, 1/n, cxpb)
        evalue = evaluate(offspring)
        tupleVal = tuple(evalue[0])          
        pop.append((offspring,evalue))
     
        if evalue[1] and not tupleVal in represented:  # new element on the front is found
            cover_count = cover_count+1
            represented.add(tupleVal)
                
            
        values = np.array([ev[0] for ind,ev in pop])    
        
           
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = values)
        # ndf: array of lists, where the i th list contains the indices of indivuals with rank i
        # ndr: 1d array, the i th entry is the rank of the i th indivual?
        
                
        critical_rank = ndf[-1]
        critical_rank_values = values[critical_rank]
        
        hv = hypervolume(critical_rank_values)
        least_contribution = hv.least_contributor(ref_point) 
        to_delete = critical_rank[least_contribution]
            
        del pop[to_delete]
        
                    
                
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
    pop_factor = float(sys.argv[6])
    verb = 0 < int(sys.argv[7])
    NGEN = int(sys.argv[8])
    seed = int(sys.argv[9])
    
    random.seed(seed)
    np.random.seed(seed)
    
    gen, pop = simulate(M, BLOCK_SIZE, cxpb, k, benchmark=benchmark, pop_factor=pop_factor, verb=verb, NGEN=NGEN)
    print(f"{benchmark},{M},{BLOCK_SIZE},{k},{cxpb},{gen},{pop_factor},{NGEN},{seed}")
    