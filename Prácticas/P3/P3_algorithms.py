from algorithms import *
import cProfile, pstats, io

np.random.seed(2022)

MAX_ITERATIONS = 200

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
        w = localSearch(ini_weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15)
        currentF = f(w, data[trainIndex], classes[trainIndex])

        if(currentF > bestF):
            weights = np.copy(w)
            bestF = currentF

        convergence.append(bestF)
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
def simulated_annealing(data,classes,trainIndex,testIndex): 
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
    M = round(MAX_ITERATIONS/max_neighbours)

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
    
    #DUDAS: PARA EL OUTER LOOP QUE CONDICIONES  HACE FALTA METER. LA DEL Nº DE ENFRIAMIENTOS ES NECESARIA ?
    # while num_successes > 0 and currentTemperature > finalTemperature and coolings < M: 
    while num_successes != 0 and iters <= MAX_ITERATIONS and currentTemperature > finalTemperature: 
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
            # AQUI HAY UN BAILE DE SIGNOS - QUE NO ENTIENDO PARA LA EXPONENCIAL
            if diff > 0 or np.random.uniform(0.0,1.0) <= np.exp(diff / currentTemperature):
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

    return t_clas_ret, t_red_ret, convergence

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

    # We apply local search to initial solution
    weights = localSearch(weights, data[trainIndex], classes[trainIndex])
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
        weights = localSearch(weights, data[trainIndex], classes[trainIndex], MAX_ITERATIONS / 15)
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

    # We apply local search to initial solution
    
    # REVISAR ESTAS LLAMADAS A LOCAL SEARCH. NO DEBERIA DE HACERSE, SI YA SE LLEGA A 15 ITERACIONES DENTRO DEL BUCLE

    # weights = localSearch(weights, data[trainIndex], classes[trainIndex])
    currentF = f(weights, data[trainIndex], classes[trainIndex])
    bestF = currentF
    best_weights = weights.copy()

    # Array for convergence values
    convergence = []
    convergence.append(bestF)
    
    # Number of mutations to make
    mutations = range(round(0.1 * data.shape[1]))

    # We initialize both parameters for cooling condition
    max_neighbours = 10 * len(weights)
    max_successes = 0.1 * max_neighbours

    # Number of coolings to make
    M = round(15000/max_neighbours)

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

    # Outer loop
    iters = 0
    while iters < 15: 
        w = weights.copy()
        # Mutations 
        for i in mutations:
            # We mute a random weight
            index = np.random.randint(0,data.shape[1]-1)
            # HAY QUE CAMBIAR LA MUTACION. LA SIGMA AHORA ES 0.4 NO 0.3
            w = mute(w, index, 0.4)

        # We apply local search to the current weights array
        w = localSearch(w, data[trainIndex], classes[trainIndex])
        newF = f(w, data[trainIndex], classes[trainIndex])

        # Acceptance criterion
        diff = newF - currentF
        # AQUI HAY UN BAILE DE SIGNOS - QUE NO ENTIENDO PARA LA EXPONENCIAL
        if diff > 0 or np.random.uniform(0.0,1.0) <= np.exp(diff / currentTemperature):
            currentF = newF
            weights = w.copy()
            if currentF > bestF:
                bestF = currentF
                best_weights = weights

        currentTemperature = currentTemperature / (1+(beta * currentTemperature))
        iters += 1
        # weights = best_weights.copy()
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