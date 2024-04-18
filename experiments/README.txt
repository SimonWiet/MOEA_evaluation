There are three kinds of files attached. 

 - .csv-Files (starting with "exp_") hold the experimental results for the respective algorithm. 
   These are comma-separated-values data files, where each row corresponds to one run of the algorithm. 
   The columns are describe
     - the tested benchmark, "OMM", "COCZ", "LOTZ", or "OJZJ" (benchmark)
     - number of blocks, that is half the number of objectives (M)
     - number of bits per block, that is n/M except for COCZ, where only the uncooperative part is considered, so n/(2M) (blockSize)
     - for OJZJ, the parameter k. 0 for other benchmarks (k)
     - Probability with which crossover was applied, here always 0 (cxpb)
     - the actual result, that is, the number of iterations until full Pareto front was sampled (gen)
     - the number of iterations after which the run would be abborted (for valid results we have gen < NGEN) (NGEN)
     - the seed used for randomness generators (seed)

   The SMS-EMOA results have one additional column
     - the population size, given as a factor to be multiplied with the size of the largest set of mutually incomparable solutions (pop_factor)
  
   The NSGA results have two additional columns
     - the maximum number of covered objective values on the Pareto front in any iteration before the NGENth generation (max_cover)
     - a list where the ith entry corresponds to the number of covered objective values on the Pareto front in the (i*S)th iteration, where S is the size of the Pareto front (cover_hist)

 - .py-Files one for each algorithm, contains the python code to run this algorithm on the benchmarks. Output corresponds to a line in the csv file. 
   Call as
	 python3 {algo}.py {M} {blockSize} {cxpb} {k} {benchmark} {verb} {NGEN} {seed}  
   Where the parameters are interpreted as explained for the .csv files and verb is an integer such that if verb > 0 intermediate status of the optimization procces is printed.
   The execution of nsga.py and smsemoa.py require additonal parameters on the population size and NSGA variant. For the provided experiments, the NSGA-II was employed with a population 4 times the size of the Pareto front and the SMS-EMOA with a population the size of the Pareto front.

 - a .ipynb-File that includes the jupyter notebook used to generate the plots.
   Whenever the main paper or supplementary material speak of X runs of an algorithm in a setting, this corresponds to a run with each of the seeds 0, 1, ..., X-1

