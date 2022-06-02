from algorithms import *
from genetics import *
import copy

sizeAM = 10
iters_softBL_AM = 10

def low_localSearch(data,classes,trainIndex,chromosome):
    bestf = chromosome.fitness
    n = len(chromosome.w)
    it = 0

    while(it < 2*n):
        k = np.random.choice(range(n))
        mutChrom = copy.deepcopy(chromosome)
        mutChrom.w = mute(mutChrom.w,k)
        mutChrom.t_class = classification_rate(mutChrom.w,data[trainIndex],classes[trainIndex])
        mutChrom.t_red = reduction_rate(mutChrom.w)
        mutChrom.fitness = 0.5 * (mutChrom.t_class + mutChrom.t_red) 
        it += 1
        if mutChrom.fitness > bestf:
            chromosome = mutChrom
            bestf = mutChrom.fitness

    return it,chromosome

"""General AM function.

    Returns:
        It returns both classification and reduction rate

    Parameters:
    data    -- dataset features
    classes -- dataset targets
    trainIndex
    testIndex
    cross_operator
    typeMemetic
    
    """
def AM(data,classes,trainIndex,testIndex,cross_operator,typeMemetic):
    population = []
    generation = 1
    bestFitness = []

    # We start by generating the population
    range_sizeAM = range(sizeAM)
    for i in range_sizeAM:
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))

    # First size_AM iterations
    it = sizeAM
    
    # Number of chromosomes to be crossed
    to_be_crossed = int(PCROSS*(len(population)/2))
    range_to_be_crossed = range(to_be_crossed*2)
    
    # Number of chromosomes to mutate. Minimum is 1
    to_mutate = max(int(PMUT_INDIVIDUO*len(population)),1)
    range_to_mutate = range(to_mutate)

    # Stop criterion: max_iteration evaluations
    while it < 15000:

        new_population = []
        
        # Calculating the chromosome with highest value of fitness        
        fitness = [pop.fitness for pop in population]
        bpIndex = np.argmax(fitness)     
        
        #Selection operator
        range_len_pop = range(len(population))
        new_population = [selection(population) for i in range_len_pop]

        #Cross operator
        for k in range_to_be_crossed:
            i1 = randint(0,len(new_population)-1)
            i2 = randint(0,len(new_population)-1)
            h1,h2 = cross_operator(new_population[i1],new_population[i2],data[trainIndex],classes[trainIndex])
            new_population[i1] = h1
            new_population[i2] = h2
            k += 1
            it +=2
 
        muted_population = new_population
        #We use possibilities to avoid repetition of value
        possibilities = []
        possibilities = [k for k in range_len_pop]
            
        #Mutation operator
        for k in range_to_mutate:
            indexPossib = np.random.randint(0,len(possibilities)-1)
            indexChromosome = possibilities[indexPossib]
            indexGen = np.random.randint(0,len(population[0].w)-1)
            new_C = Chromosome(data[trainIndex],classes[trainIndex],mute(muted_population[indexChromosome].w,indexGen))
            muted_population[indexChromosome] = new_C
            possibilities.pop(indexPossib)
            it +=1
            
    
        new_population = np.copy(muted_population)


        #Find new best
        # Calculating the chromosome with highest value of fitness        
        fitness = [pop.fitness for pop in new_population]
        currentBestIndex = np.argmax(fitness)

        #Elitism       
        if new_population[currentBestIndex].fitness < population[bpIndex].fitness:
            # Calculating the chromosome with lowest value of fitness        
            fitness = [pop.fitness for pop in new_population]
            newWorst = np.argmin(fitness)
            new_population = np.delete(new_population,newWorst)
            new_population = np.append(new_population, population[bpIndex])
           
                                  
        population = np.copy(new_population)
        
        # Check if we have to apply soft local search
        if (generation % iters_softBL_AM) == 0:
            s,population = typeMemetic(population,data,classes,trainIndex)
            it +=s

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

"""AM - (10, 1.0). We apply BL to each of the chromosomes

    Returns:
        It returns both new population, and current evaluations of the objective function       

    Parameters:
    it      -- current evaluations of the objective function
    data    -- dataset features
    classes -- dataset targets
    trainIndex
    
    """
def am1(population,data,classes,trainIndex):
    it = 0
    new_population = []
    for c in population:
        s,newC = low_localSearch(data,classes,trainIndex,c)
        it += s
        new_population.append(newC)

    return it,new_population

"""AM - (10, 0.1). We apply BL to a random subset of chromosomes. As probability is 0.1, we just
   have to select one random chromosome. 

    Returns:
        It returns both new population, and current evaluations of the objective function       

    Parameters:
    it      -- current evaluations of the objective function
    data    -- dataset features
    classes -- dataset targets
    trainIndex
    
    """
def am2(population,data,classes,trainIndex):
    k = np.random.choice(range(sizeAM-1))
    it,population[k] = low_localSearch(data,classes,trainIndex,population[k])
    return it, population

"""AM - (10, 0.1mej). We apply BL to the chromosome whose fitness value is the highest

    Returns:
        It returns both new population, and current evaluations of the objective function       

    Parameters:
    it      -- current evaluations of the objective function
    data    -- dataset features
    classes -- dataset targets
    trainIndex
    
    """
def am3(population,data,classes,trainIndex):
    fitness = [pop.fitness for pop in population]
    best = np.argmax(fitness)

    it,population[best] = low_localSearch(data,classes,trainIndex,population[best])

    return it,population