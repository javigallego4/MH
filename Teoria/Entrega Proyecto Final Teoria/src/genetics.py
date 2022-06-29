from algorithms import *

POPULATION_SIZE = 30
PCROSS= 0.7
PMUT_INDIVIDUO  = 0.1
MAX_EVALUACIONES = 15000
np.random.seed(2022)

class Chromosome:
    def __init__(self,data,classes,weights = []):
        if len(weights) == 0:
            self.w = np.random.uniform(0.0,1.0,len(data[0]))
        else:
            self.w = np.copy(weights)

        self.t_class = classification_rate(self.w,data,classes)
        self.t_red = reduction_rate(self.w)                    
        self.fitness = 0.5 * (self.t_class + self.t_red)
        
"""BLX-0.3 Cross.

    Returns:
        It returns two new descendants to whom we've applied BLX-0.3 cross        

    Parameters:
    c1      -- first chromosome
    c2      -- second chromosome
    data    -- dataset features
    classes -- dataset targets
    
    """
def BLX(c1,c2,data,classes):
    cmax  = max([max(c1.w),max(c2.w)])
    cmin  = min([min(c1.w),min(c2.w)])
    l     = cmax - cmin
    #interval is [a,b]
    a     = cmin - l*0.3
    b     = cmax + l*0.3

    H1 = np.random.uniform(a,b,len(c1.w))
    H2 = np.random.uniform(a,b,len(c2.w))

    H1[H1<0.0] = 0.0
    H1[H1>1.0] = 1.0

    H2[H2<0.0] = 0.0
    H2[H2>1.0] = 1.0

    return Chromosome(data,classes,H1),Chromosome(data,classes,H2)

"""Arithmetic Cross.

    Returns:
        It returns two new descendants to whom we've applied arithmetic cross        

    Parameters:
    c1      -- first chromosome
    c2      -- second chromosome
    data    -- dataset features
    classes -- dataset targets
    
    """
def arithmeticCross(c1,c2,data,classes):
    alpha = np.random.uniform(0.0,1.0)
    beta = np.random.uniform(0.0,1.0)

    H1 = np.array(c1.w) * alpha + np.array(c2.w) * (1-alpha)
    H2 = np.array(c1.w) * beta + np.array(c2.w) * (1-beta)

    H1[H1<0.0] = 0.0
    H1[H1>1.0] = 1.0

    H2[H2<0.0] = 0.0
    H2[H2>1.0] = 1.0

    return Chromosome(data,classes,H1),Chromosome(data,classes,H2)
    
"""Selection Function - Binary Tournament.

    Returns:
        It returns index of the chromosome whose fitness is greater

    Parameters:
    pop -- population
    
    """
def selection(pop):
    i1 = randint(0,len(pop)-1)
    i2 = randint(0,len(pop)-1)
    if pop[i1].fitness > pop[i2].fitness:
        return pop[i1]
    else:
        return pop[i2]

"""Generational genetic algorithm.

    Returns:
        It returns both classification and reduction rate        

    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex
    cross_operator -- type of crossing
    
    """
def AGG(data,classes,trainIndex,testIndex,cross_operator):
    generation = 1
    population = []
    fitness = []
    bestFitness = []

    # We start by generating the population
    range_POPULATION_SIZE = range(POPULATION_SIZE)
    for i in range_POPULATION_SIZE:
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))       

    # first POPULATION_SIZE iterations
    it  = POPULATION_SIZE

    # Pairs of chromosomes to be crossed
    to_be_crossed = int(PCROSS* (len(population)/2))
    range_to_be_crossed = range(to_be_crossed*2)

    # Number of chromosomes to mutate. Minimum is 1
    to_mutate = max(int(PMUT_INDIVIDUO*len(population)),1)
    range_to_mutate = range(to_mutate)

    # Stop criterion: max_iteration evaluations
    while it < MAX_EVALUACIONES:

        s = time.time()
        new_population = []
        
        # Calculating the chromosome with highest value of fitness        
        fitness = [pop.fitness for pop in population]
        bpIndex = np.argmax(fitness)      
        
        # Selection operator
        range_len_pop = range(len(population))
        new_population = [selection(population) for i in range_len_pop]

        # Cross operator
        for k in range_to_be_crossed:
            i1 = randint(0,len(new_population)-1)
            i2 = randint(0,len(new_population)-1)
            h1,h2 = cross_operator(new_population[i1],new_population[i2],data[trainIndex],classes[trainIndex])
            new_population[i1] = h1
            new_population[i2] = h2
            k += 1
            it +=2
 
        muted_population = new_population
        
        # We use possibilities to avoid repetition of value
        possibilities = []          
        possibilities = [k for k in range_len_pop]
        
        # Mutation operator
        for k in range_to_mutate:
            indexPossib = np.random.randint(0,len(possibilities)-1)
            indexChromosome = possibilities[indexPossib]
            indexGen = np.random.randint(0,len(population[0].w)-1)
            new_C = Chromosome(data[trainIndex],classes[trainIndex],mute(muted_population[indexChromosome].w,indexGen))
            muted_population[indexChromosome] = new_C
            possibilities.pop(indexPossib)
            it +=1
            
        new_population = np.copy(muted_population)
        range_new_pop = range(len(new_population))        

        # Calculating the chromosome with highest value of fitness        
        fitness = [pop.fitness for pop in new_population]
        currentBestIndex = np.argmax(fitness)
        
        # Elitism       
        if new_population[currentBestIndex].fitness < population[bpIndex].fitness:
            # Calculating the chromosome with lowest value of fitness        
            fitness = [pop.fitness for pop in new_population]
            newWorst = np.argmin(fitness)
            new_population = np.delete(new_population,newWorst)
            new_population = np.append(new_population, population[bpIndex])
                                  
        population = np.copy(new_population)
        # Calculating the chromosome with highest value of fitness        
        fitness = [pop.fitness for pop in new_population]
        currentBestIndex = np.argmax(fitness)
        bestFitness.append(population[currentBestIndex].fitness)
        
        generation += 1
    

    finalW = population[currentBestIndex].w
    trainD = np.copy(data[trainIndex])*finalW
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*finalW
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(finalW)
    
    return t_clas_ret, t_red_ret, bestFitness

"""Stationary genetic algorithm.

    Returns:
        It returns both classification and reduction rate

    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex
    cross_operator -- type of crossing
    
    """
def AGE(data,classes,trainIndex,testIndex,cross_operator) :
    generation = 1
    population = []
    fitness = []
    bestFitness = []

    # We start by generating the population
    range_POPULATION_SIZE = range(POPULATION_SIZE)
    for i in range_POPULATION_SIZE:
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))
    
    # First POPULATION_SIZE iterations
    it   = POPULATION_SIZE

    # Number of gens to mutate. Minimum is 1, as we want diversity    
    to_mutate = max(int(PMUT_INDIVIDUO*len(population)),1)
    range_to_mutate = range(to_mutate)

    # Stop criterion: max_iteration evaluations
    while it < MAX_EVALUACIONES:                        
        # Both two parents of the AGE
        new_parents = [selection(population),selection(population)]

        # Cross operator
        new_parents[0],new_parents[1] = cross_operator(new_parents[0],new_parents[1],data[trainIndex],classes[trainIndex])                
        it +=2
        
        # Mutation operator
        # for k in range_to_mutate:
        #     idx = np.random.randint(0,1)
        #     indexGen = np.random.randint(0,len(population[0].w)-1)
        #     new_C = Chromosome(data[trainIndex],classes[trainIndex],mute(new_parents[idx].w,indexGen))
        #     new_parents[idx] = new_C
        #     it +=1 
        for k in range(2):
            prob = random.random()
            if prob < PMUT_INDIVIDUO:
                indexGen = np.random.randint(0,len(population[0].w)-1)
                new_C = Chromosome(data[trainIndex],classes[trainIndex],mute(new_parents[k].w,indexGen))
                new_parents[k] = new_C
                it +=1             
    
        # Replacement. The process is the following: appending them both, 
        # sorting the population by its fitness values, erasing the two worst chromosomes.
        population = np.append(population,new_parents[0])
        population = np.append(population,new_parents[1])
        population = sorted(population, key = lambda x : x.fitness)
        population = np.delete(population,0)
        population = np.delete(population,0)
        population = shuffle(population,random_state = 0)


        fitness = [pop.fitness for pop in population]
        currentBestIndex = np.argmax(fitness)
        bestFitness.append(population[currentBestIndex].fitness)

        generation += 1


    finalW = population[currentBestIndex].w
    trainD = np.copy(data[trainIndex])*finalW
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*finalW
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(finalW)

    return t_clas_ret, t_red_ret, bestFitness