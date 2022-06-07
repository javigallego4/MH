from algorithms import *
import cProfile, pstats, io

np.random.seed(123456789)

MAX_ITERATIONS = 15000

""" A decorator that uses cProfile to profile a function """
def profile(fnc): 
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

"""Basic multi-start search.

    Returns: 
        It returns value of both classification and reduction rate. 
    
    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex

    """
def bmb(data,classes,trainIndex,testIndex):
    # We start by generating an initial random solution
    ini_weights = np.random.uniform(0.0,1.0,data.shape[1])
    bestF = f(ini_weights, data[trainIndex], classes[trainIndex])

    # Array for convergence values. 
    convergence = []
    convergence.append(bestF)

    iters = range(15)
    for i in iters: 
        w, convergence = localSearch(ini_weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15, convergence, bestF)
        currentF = f(w, data[trainIndex], classes[trainIndex])

        if(currentF > bestF):
            weights = np.copy(w)
            bestF = currentF

        # convergence.append(bestF)
        ini_weights = np.random.uniform(0.0,1.0,data.shape[1])

    trainD = np.copy(data[trainIndex])*weights
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*weights
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(weights)

    return t_clas_ret, t_red_ret, convergence
        
"""Simulated annealing.

    Returns: 
        It returns value of both classification and reduction rate. 
    
    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex

    """
def simulated_annealing(data,classes,trainIndex,testIndex, max_iterations = MAX_ITERATIONS, weights = []): 
    if len(weights) == 0: 
        # We start by generating an initial random solution
        weights = np.random.uniform(0.0,1.0,data.shape[1])
    
    currentF = f(weights, data[trainIndex], classes[trainIndex])
    bestF = currentF
    best_weights = weights.copy()

    # Array for convergence values
    convergence = []
    convergence.append(currentF)

    # We initialize both parameters for cooling condition
    max_neighbours = 10 * len(weights)
    max_successes = 0.1 * max_neighbours

    # Number of coolings to make
    M = round(max_iterations/max_neighbours)

    # Both initial and final temperatures
    initialTemperature = (0.3 * currentF)/(-np.log(0.3))
    finalTemperature = 0.001

    # Final temperature must be lower than the initial one. 
    while finalTemperature > initialTemperature: 
        finalTemperature *= 0.001

    # We calculate now beta value
    beta = (initialTemperature - finalTemperature)/(M*initialTemperature*finalTemperature)

    # We initialize current temperaute
    currentTemperature = initialTemperature

    num_successes = 1
    # neighbours = 0
    coolings = 0
    iters = 1

    # Outer loop
    
    # ¿¿ aqui quitar lo de curretnTemperature > final porque sino se sale mucho antes ??
    #DUDAS: PARA EL OUTER LOOP QUE CONDICIONES  HACE FALTA METER. LA DEL Nº DE ENFRIAMIENTOS ES NECESARIA ?
    # while num_successes > 0 and currentTemperature > finalTemperature and coolings < M: 
    while num_successes != 0 and iters <= max_iterations and currentTemperature > finalTemperature: 
        num_successes = 0
        neighbours = 0

        # Inner loop (cooling)
        while num_successes < max_successes and neighbours < max_neighbours: 
            w = weights.copy()

            # We mute a random weight
            index = np.random.randint(0,len(w)-1)
            w = mute(w, index)

            newF = f(w, data[trainIndex], classes[trainIndex])
            iters += 1
            
            # Acceptance criterion            
            diff = newF - currentF
            # Nuevo fitness al actual, o se da la probabilidad
            if diff > 0 or np.random.uniform(0.0,1.0) <= np.exp( diff / currentTemperature):
                currentF = newF
                weights = w.copy()
                num_successes += 1
                if currentF > bestF: 
                    bestF = currentF
                    best_weights = weights

            neighbours += 1
            convergence.append(bestF)

        # We calculate current temperature
        currentTemperature = currentTemperature / (1+(beta * currentTemperature))
        coolings +=1

    trainD = np.copy(data[trainIndex])*best_weights
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*best_weights
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(best_weights)

    print('Iteraciones Enfriamiento Simulado', iters)

    if max_iterations == MAX_ITERATIONS:
        return t_clas_ret, t_red_ret, convergence
    # Para el híbrido
    else:
        return best_weights

"""Iterative local search.

    Returns: 
        It returns value of both classification and reduction rate. 
    
    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex

    """
def ils(data,classes,trainIndex,testIndex):
    # We start by generating an initial random solution
    weights = np.random.uniform(0.0,1.0,data.shape[1])  

    # bestF = f(weights, data[trainIndex], classes[trainIndex])

    # Array for convergence values. We initialize it to [0] because of LocalSearch function declaration. 
    # In order to keep appending values to this array. 
    convergence = [0]
    # convergence.append(bestF)  

    # We apply local search to initial solution
    weights, convergence = localSearch(weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15, convergence)
    bestF = f(weights, data[trainIndex], classes[trainIndex])
    best_weights = weights.copy()
    
    # We erase the first 0 value we appended at first. 
    convergence = convergence[1:]

    # # Array for convergence values
    # convergence = []
    # convergence.append(bestF)

    # Number of mutations to make
    mutations = range(round(0.1 * data.shape[1]))

    # Outer loop
    iters = range(14)
    for i in iters: 
        # Mutations 
        for i in mutations:
            # We mute a random weight
            index = np.random.randint(0,data.shape[1]-1)
            weights = mute(weights, index, 0.4)

        # We apply local search to the current weights array
        weights, convergence = localSearch(weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15, convergence, bestF)
        currentF = f(weights, data[trainIndex], classes[trainIndex])

        if currentF > bestF:
            bestF = currentF
            best_weights = weights

        weights = best_weights.copy()
        # convergence.append(bestF)

    trainD = np.copy(data[trainIndex])*best_weights
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*best_weights
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(best_weights)

    return t_clas_ret, t_red_ret, convergence
    

"""Hybrid between ILS - ES.

    Returns: 
        It returns value of both classification and reduction rate. 
    
    Parameters:
    data          -- dataset features
    classes       -- dataset targets
    trainIndex
    testIndex

    """
def hybrid_ils_es(data,classes,trainIndex,testIndex):
    # We start by generating an initial random solution
    weights = np.random.uniform(0.0,1.0,data.shape[1])    

    # convergence = [0]

    # We apply local search to initial solution
    weights = simulated_annealing(data, classes, trainIndex,testIndex, MAX_ITERATIONS / 15, weights)
    bestF = f(weights, data[trainIndex], classes[trainIndex])
    best_weights = weights.copy()

    # Array for convergence values
    convergence = []
    convergence.append(bestF)

    # Number of mutations to make
    mutations = range(round(0.1 * data.shape[1]))

    # Outer loop
    iters = range(14)
    for i in iters: 
        # Mutations 
        for i in mutations:
            # We mute a random weight
            index = np.random.randint(0,data.shape[1]-1)
            weights = mute(weights, index, 0.4)

        # We apply local search to the current weights array
        # weights = localSearch(weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15)
        weights = simulated_annealing(data, classes, trainIndex,testIndex, MAX_ITERATIONS / 15, weights)
        currentF = f(weights, data[trainIndex], classes[trainIndex])

        if currentF > bestF:
            bestF = currentF
            best_weights = weights

        weights = best_weights.copy()
        convergence.append(bestF)

    trainD = np.copy(data[trainIndex])*best_weights
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*best_weights
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)
    t_red_ret   = reduction_rate(best_weights)

    return t_clas_ret, t_red_ret, convergence