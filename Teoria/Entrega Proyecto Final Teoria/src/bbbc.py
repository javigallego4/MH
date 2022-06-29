import numpy as np
import random
from algorithms import mute, classification_rate, reduction_rate
from genetics import *
from P3_algorithms import *
np.random.seed(2022)

ELITE_POOL_SIZE = 10
NUM_NEIGHBOURS = 5

class StellarObject:
    def __init__(self,data,classes,weights = []):
        if len(weights) == 0:
            self.w = np.random.uniform(0.0,1.0,len(data[0]))
        else:
            self.w = np.copy(weights)

        self.t_class = classification_rate(self.w,data,classes)
        self.t_red = reduction_rate(self.w)                    
        self.mass = 0.5 * (self.t_class + self.t_red)

"""Neighbours Generation.

    Returns:
        It returns neighbour with highest mass value        

    Parameters:
    st            -- Stellar Object
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex
    
    """
def neighbours(st,data,classes, trainIndex, testIndex, cmax, cmin, convergence):    
    neighbours = []
    sigma = max([0.05, cmax-cmin])

    # We generate NUM_NEIGHBOURS neighbours for Stellar Object st. 
    size = range(NUM_NEIGHBOURS)
    for i in size:
        for j in range(int(len(st.w)*sigma)):
            weights = np.copy(st.w)
            index = random.randint(0,len(weights)-1)
            weights = mute(weights, index)            
            # new_st = StellarObject(data[trainIndex],classes[trainIndex],mute(st.w, index))
        new_st = StellarObject(data[trainIndex],classes[trainIndex],weights)
        neighbours.append(new_st)
        if new_st.mass > convergence[-1]: convergence.append(new_st.mass)
        else: convergence.append(convergence[-1])

    # We return best generated neigbour 
    mass = [n.mass for n in neighbours]
    bestIndex = np.argmax(mass)
    return neighbours[bestIndex] 


"""Big Bang - Big Crunch algorithm.

    Returns:
        It returns both classification and reduction rate        

    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex
    
    """
def BBBC(data,classes,trainIndex,testIndex): 
    population = []
    seguir = True
    convergence = []

    # Step 1: We generate the initial random solutions
    range_POPULATION_SIZE = range(POPULATION_SIZE)
    for i in range_POPULATION_SIZE:
        population.append(StellarObject(data[trainIndex],classes[trainIndex]))           
        if i != 0: 
            if population[i].mass > convergence[i-1]: convergence.append(population[i].mass)
            else: convergence.append(convergence[i-1])
        else: convergence.append(population[0].mass)

    # We sort our population, depending on mass values. 
    population = sorted(population, reverse = True, key = lambda x : x.mass)

    # We calculate the mass center and elite pool
    centre_mass = np.copy(population[0])
    elite_pool = np.copy(population[:ELITE_POOL_SIZE])

    cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass

    # Stop criterion: max_iteration evaluations
    it = POPULATION_SIZE 
    while it < MAX_EVALUACIONES:
        seguir = True
        new_tam = 30
        while seguir == True: 
            new_population = []

            # Step 2: We generate N neighbours per each of the elements in the population
            length = range(new_tam)
            for i in length:
                best_neighbour = neighbours(population[i], data, classes, trainIndex, testIndex, cmax, cmin, convergence) 
                new_population.append(best_neighbour)
            
            it += new_tam * NUM_NEIGHBOURS 

            # Step 3 Find centre of mass
            new_population = sorted(new_population, reverse = True, key = lambda x : x.mass)            
            centre_mass = new_population[0] 

            # Step 5: We update elite pool
            if centre_mass.mass > elite_pool[-1].mass: 
                elite_pool = np.delete(elite_pool, ELITE_POOL_SIZE-1)
                elite_pool = np.append(elite_pool, centre_mass)

                elite_pool = sorted(elite_pool, reverse = True, key = lambda x : x.mass)
                cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass
                cmax_reduccion = elite_pool[0].t_red

            # Step 6: We reduce current population size
            if (len(new_population) != 6):
                new_population = new_population[:len(new_population) - 6]
                new_tam = len(new_population)
            else:
                new_population = new_population[0]
                seguir = False

        if it + (POPULATION_SIZE - ELITE_POOL_SIZE) < MAX_EVALUACIONES:
            # GENERAMOS LA NUEVA POBLACION A PARTIR DE LA ELITE POOL
            population = []
            population[:ELITE_POOL_SIZE] = np.copy(elite_pool)

            r = range(int((POPULATION_SIZE - ELITE_POOL_SIZE) / ELITE_POOL_SIZE))
            for i in range(ELITE_POOL_SIZE):
                for j in r: 
                    index = random.randint(0,len(elite_pool[i].w)-1)
                    population.append(StellarObject(data[trainIndex],classes[trainIndex],mute(np.copy(elite_pool[i].w), index)))

                    # HAY QUE CONTAR ESTAS EVALUACIONES !!!!!!!!!!!
                    if population[-1].mass > elite_pool[-1].mass: 
                        elite_pool = np.delete(elite_pool, ELITE_POOL_SIZE-1)
                        elite_pool = np.append(elite_pool, centre_mass)
                        elite_pool = sorted(elite_pool, reverse = True, key = lambda x : x.mass)
                        cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass
                        convergence.append(cmax)

            it += POPULATION_SIZE - ELITE_POOL_SIZE

    finalW = elite_pool[0].w
    trainD = np.copy(data[trainIndex])*finalW
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*finalW
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)

    return t_clas_ret, elite_pool[0].t_red, convergence

                               
"""Big Bang - Big Crunch Hybrid algorithm.

    Returns:
        It returns both classification and reduction rate        

    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex
    
    """
def BBBC_LocalSearch(data,classes,trainIndex,testIndex): 
    convergence = []
    population = []
    seguir = True

    # Step 1: We generate the initial random solutions
    range_POPULATION_SIZE = range(POPULATION_SIZE)
    for i in range_POPULATION_SIZE:
        population.append(StellarObject(data[trainIndex],classes[trainIndex])) 
        if i != 0: 
            if population[i].mass > convergence[i-1]: convergence.append(population[i].mass)
            else: convergence.append(convergence[i-1])
        else: convergence.append(population[0].mass)          

    # We sort our population, depending on mass values. 
    population = sorted(population, reverse = True, key = lambda x : x.mass)

    # We calculate the mass center and elite pool
    centre_mass = np.copy(population[0])
    elite_pool = np.copy(population[:ELITE_POOL_SIZE])

    cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass

    # Stop criterion: max_iteration evaluations
    it = POPULATION_SIZE 
    while it < MAX_EVALUACIONES:
        seguir = True
        new_tam = 30
        while seguir == True: 
            new_population = []

            # Step 2: We generate N neighbours per each of the elements in the population
            length = range(new_tam)
            for i in length:
                best_neighbour = neighbours(population[i], data, classes, trainIndex, testIndex, cmax, cmin, convergence) 
                new_population.append(best_neighbour)
            
            it += new_tam * NUM_NEIGHBOURS 

            # Step 3 Find centre of mass
            new_population = sorted(new_population, reverse = True, key = lambda x : x.mass)            
            weights, convergence, aux = localSearch(np.copy(new_population[0].w), data[trainIndex], classes[trainIndex], 100, convergence, np.copy(convergence[-1]))
            it += aux
            centre_mass = StellarObject(data[trainIndex],classes[trainIndex], weights)

            # Step 5: We update elite pool
            if centre_mass.mass > elite_pool[-1].mass:
                elite_pool = np.delete(elite_pool, ELITE_POOL_SIZE-1)
                elite_pool = np.append(elite_pool, centre_mass)
                elite_pool = sorted(elite_pool, reverse = True, key = lambda x : x.mass)
                cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass

            # Step 6: We reduce current population size
            if (len(new_population) != 6):
                new_population = new_population[:len(new_population) - 6]
                new_tam = len(new_population)
            else:
                new_population = new_population[0]
                seguir = False

        if it + (POPULATION_SIZE - ELITE_POOL_SIZE) < MAX_EVALUACIONES:
            population = []
            population[:ELITE_POOL_SIZE] = np.copy(elite_pool)

            r = range(int((POPULATION_SIZE - ELITE_POOL_SIZE) / ELITE_POOL_SIZE))
            for i in range(ELITE_POOL_SIZE):
                for j in r: 
                    index = random.randint(0,len(elite_pool[i].w)-1)
                    population.append(StellarObject(data[trainIndex],classes[trainIndex],mute(np.copy(elite_pool[i].w), index)))

                    if population[-1].mass > elite_pool[-1].mass: 
                        elite_pool = np.delete(elite_pool, ELITE_POOL_SIZE-1)
                        elite_pool = np.append(elite_pool, centre_mass)
                        elite_pool = sorted(elite_pool, reverse = True, key = lambda x : x.mass)
                        cmax, cmin = elite_pool[0].mass, elite_pool[-1].mass
                        convergence.append(cmax)


            it += POPULATION_SIZE - ELITE_POOL_SIZE

    finalW = elite_pool[0].w
    trainD = np.copy(data[trainIndex])*finalW
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*finalW
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)

    return t_clas_ret, elite_pool[0].t_red, convergence                               