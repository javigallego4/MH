import numpy as np
import operator
# from scipy.spatial import distance
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.model_selection import LeaveOneOut
from numpy.random import normal
import random
from random import randint
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numba


"""Distance between numerical features.

    Returns: 
        It returns euclidean distance between e1 and e2
    
    Parameters:
    e1 -- 1st point
    e2 -- 2nd point

    """
def distance(e1, e2):
    return np.sqrt(np.sum((e1-e2)**2, axis=1))

"""1-Nearest Neighbours Algorithm (1-NN).

    Returns: 
        It returns nearest neighbour
    
    Parameters:
    features -- dataset features values
    targets -- targets values
    example -- element in test section

    """
def one_nn(features, targets, example):
    return targets[np.argmin(distance(features, example))]          
    
"""Greedy Relief Algorithm.

    Returns: 
        It returns weights array
    
    Parameters:
    features -- dataset features 
    targets -- targets values    

    """
def greedyRelief(features,targets):
    # Initialize weights array w to 0
    w = np.zeros(shape=(features.shape[1],))

    # Calculate every distance 
    distances = squareform(pdist(features))
    np.fill_diagonal(distances, np.infty)

    # Optimization
    i = 0
    entryRange = features.shape[0]
    
    # For every sample in the train set: 
    #for i in range(0, features.shape[0]):    
    while i < entryRange:
        # enemies, friends indexes
        en_indices = targets != targets[i]
        fr_indices = targets == targets[i]

        enemies = features[en_indices]
        friends = features[fr_indices]
        
        # Identify both nearest enemy and friend
        closest_friend = np.argmin(distances[fr_indices, i])
        closest_enemy = np.argmin(distances[en_indices, i])

        # Update weights array W
        w = w + np.abs(features[i]-enemies[closest_enemy]) - \
            np.abs(features[i]-friends[closest_friend])
        
        i += 1

    # Truncate negative values to 0, and normalize
    w[w<0] = 0
    w = w/np.max(w)
    return w

"""Weighted 1-Nearest Neighbours.

    Returns: 
        It returns nearest neighbour, using weights array
    
    Parameters:
    weights -- weights array
    features -- dataset features values
    targets -- targets values  
    example -- element in test section  

    """
def weighted_onenn(weights,features,targets,example):
    dist = weights*(features - example)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return targets[np.argmin(dist)]

"""Reduction rate.

    Returns: 
        It returns reducted wegiths array
    
    Parameters:
    w -- weights array 

    """
def reduction_rate(w):
    return ((w[w < 0.2].shape[0])/w.shape[0])

"""Classification Rate.

    Returns: 
        It returns percentage of positive values (positive = classified in correct class)
    
    Parameters:
    w -- weights array
    features -- dataset features values 
    targets -- targets values  

    """
def classification_rate(w,features,targets):
    dataw = features*w
    classifier = KNeighborsClassifier(n_neighbors = 1, weights='uniform')
    classifier.fit(dataw,targets)
    ind_near = classifier.kneighbors(dataw,n_neighbors=2)[1][:,1]    
    #it += 1
    tasa_clas = np.mean(targets[ind_near] == targets)        

    return tasa_clas

"""Aggregate Function. Our aim is to maximize it.

    Returns: 
        It returns value for objective function
    
    Parameters:
    w -- weights array
    features -- dataset features values 
    targets -- targets values  

    """
def f(w,features,targets): 
    alpha = 0.5    
    return alpha*classification_rate(w,features,targets) +(1-alpha)*reduction_rate(w)

"""Mutation function.

    Returns: 
        It returns muted weights array
    
    Parameters:
    w -- weights array
    sigma -- sigma value
    j -- index of element we're gonna mute

    """
def mute(w,j, sigma = 0.3):
    # np.clip limits possible value
    w[j] = np.clip(w[j] + np.random.normal(scale=sigma), 0, 1)
    return w

"""Local Search Algorithm.

    Returns: 
        It returns weights array
    
    Parameters:
    ini_weights -- random initial weights (using normal distribution in [0,1])
    features -- dataset features values
    targets -- targets values

    """
def localSearch(ini_weight,features,targets, max_iterations = 15000, convergence = [], globalBest = 0):
    if len(convergence) == 0:
        return_convergence = False
    else:
        return_convergence = True

    n = features.shape[1]
    count = 0
    improve = False
    # Initial solution weights and objective function value
    weights = ini_weight
    bestF = f(ini_weight,features,targets)

    # Range() list is inmutable. Therefore, we introduce indexes
    # by ourselves
    index = []
    for i in range(0,n):
        index.append(i)
    random.shuffle(index)

    # Optimization
    actual_iter = 0
    maximum_neighbours = n*20    
    notMuted = 0   #Count if mutes is less than n*20
    
    #entryRange_i = range(0,max_iterations)
    #for i in entryRange_i:
    while actual_iter < max_iterations and notMuted < maximum_neighbours:         
        # Component that we are going to mute
        j = index[actual_iter % n]
        # Muting
        w = np.copy(weights)
        w = mute(w,j)
        newF = f(w,features,targets)
        # Acceptance criterion
        if(newF > bestF):
            bestF = newF
            weights = w
            notMuted = 0
            improve = True
        else:
            notMuted += 1

        actual_iter += 1
        # Whether there's been an improvement or every component
        # has been already muted:
        if(actual_iter % n == count or improve):
            count = actual_iter % n
            improve = False
            random.shuffle(index)
            
        if(bestF > globalBest):
            globalBest = bestF
            
        if return_convergence == True: convergence.append(globalBest)


    print('Máximas evaluaciones: ', max_iterations)
    print('Iteracion en la que se sale la LS', actual_iter)
    if return_convergence == False: return weights
    if return_convergence == True: return weights, convergence
