import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from numpy.lib import recfunctions as rfn
from statistics import *
from algorithms import *
from numpy.random import uniform
import time

dataset_ionosphere = 'Datasets/ionosphere.arff'
dataset_parkinson = 'Datasets/parkinsons.arff'
dataset_heart = 'Datasets/spectf-heart.arff'

"""Load one specific dataset.

    Returns:
        It returns both dataset features and target             

    Parameters:
    filename -- file of type .arff to load
    
    """
def load_arff(filename):
    dataset = loadarff(filename)[0]
    if filename == dataset_ionosphere:
        target = dataset['class']
        dataset_features = rfn.drop_fields(dataset, 'class').view(np.float64).reshape(dataset.shape+(-1,))
    else:
        target = dataset['Class']
        dataset_features = rfn.drop_fields(dataset, 'Class').view(np.float64).reshape(dataset.shape+(-1,))
    return dataset_features, target

"""Executes algorithms for given dataset.

    Returns:
        It returns results obtained for each of the algorithms             

    Parameters:
    dataset -- dataset name
    dataset_features -- dataset features
    dataset target -- dataset target
    
    """
def execution_results(dataset, dataset_features, dataset_target):    
    i = 0   # Indicates partition
    skf = StratifiedKFold(n_splits = 5) # cross-validation
    # Optimization
    entryRange_x = range(4)
    entryRange_y = range(5)
    dataset_1NN = [[0 for x in entryRange_x] for y in entryRange_y]
    dataset_GR = [[0 for x in entryRange_x] for y in entryRange_y]
    dataset_LS = [[0 for x in entryRange_x] for y in entryRange_y]

    print("File: ", dataset)
    split_indexes = skf.split(dataset_features,dataset_target)
    for trainIndex , testIndex in split_indexes:
        onennTime = 0
        greedyTime = 0
        localSTime = 0
        n = testIndex.shape[0]
        onennRight = 0
        greedyRight = 0
        lsRight = 0
        # Time for weights calc
        startTime = time.time()
        greedyWeights = greedyRelief(dataset_features[trainIndex],dataset_target[trainIndex])
        greedyTime += time.time() - startTime
        ini_uniform_weights = np.random.uniform(0.0,1.0,dataset_features.shape[1])
        # Time for waeights calc
        startTime = time.time()
        lsWeights = localSearch(ini_uniform_weights,dataset_features[trainIndex],dataset_target[trainIndex])
        localSTime += time.time() - startTime

        for element in testIndex:
            startTime = time.time()
            if one_nn(dataset_features[trainIndex],dataset_target[trainIndex],dataset_features[element]) == dataset_target[element]:
                onennRight +=1
            onennTime += time.time()-startTime

            startTime = time.time()
            if weighted_onenn(greedyWeights,dataset_features[trainIndex],dataset_target[trainIndex],dataset_features[element]) == dataset_target[element]:
                greedyRight +=1
            greedyTime += time.time() -startTime

            startTime = time.time()
            if weighted_onenn(lsWeights,dataset_features[trainIndex],dataset_target[trainIndex],dataset_features[element]) == dataset_target[element]:
                lsRight += 1
            localSTime += time.time() - startTime

        # Save results
        dataset_1NN[i][0] = onennRight/n
        dataset_1NN[i][1] = 0 #its always 0 in this case
        dataset_1NN[i][2] = 0.5*(onennRight/n)
        dataset_1NN[i][3] = onennTime
        dataset_GR[i][0]  = greedyRight/n
        dataset_GR[i][1]  = reduction_rate(greedyWeights)
        dataset_GR[i][2]  = (greedyRight/n+reduction_rate(greedyWeights))/2
        dataset_GR[i][3]  = greedyTime
        dataset_LS[i][0]  = lsRight/n
        dataset_LS[i][1]  = reduction_rate(lsWeights)
        dataset_LS[i][2]  = (lsRight/n+reduction_rate(lsWeights))/2
        dataset_LS[i][3]  = localSTime
        i+=1
        print("Partition ",i)
    
    return dataset_1NN, dataset_GR, dataset_LS


"""Prints out medium values.

    Parameters:
    d1_algoritmo -- algorithm for which we're gonna print medium values

    """
def print_media(d1_algoritmo):
    for j in range(3):
        lista = []
        for i in range(5):
            lista.append(d1_algoritmo[i][j])
        media = np.mean(lista)
        print("%.2f" % media, end=" - ")
    lista = []
    for i in range(5):
        lista.append(d1_algoritmo[i][3])
    media = np.mean(lista)
    if len(str(media)) == 8:
        print("%.4f" % media, end=" - ")
    else:
        print("%.5f" % media, end=" - ")

"""Prints out standard desviation values.

    Parameters:
    d_algoritmo -- algorithm for which we're gonna print std values

    """
def print_desviacion(d_algoritmo):
    for j in range(3):
        lista = []
        for i in range(5):
            lista.append(d_algoritmo[i][j])
        desviacion = np.std(lista)
        print("%.2f" % desviacion, end=" - ")
    lista = []
    for i in range(5):
        lista.append(d_algoritmo[i][3])
    desviacion = np.std(lista)
    if len(str(desviacion)) == 8:
        print("%.4f" % desviacion, end=" - ")
    else:
        print("%.5f" % desviacion, end=" - ")

"""Prints out results obtained by executing this file.

    Parameters:
    d_algoritmo -- 1st algorithm
    d2_algoritmo -- 2nd algorithm
    d3_algoritmo -- 3rd algorithm

    """
def print_out(d1_algoritmo, d2_algoritmo, d3_algoritmo):
    print("Partition  \t Ionosphere \t \t \t     Parkinsons \t \t \t Spectf-heart \t")
    print("   \tT_clas - T_red - Agr - T\t\tT_clas - T_red - Agr - T \t\tT_clas - T_red - Agr - T ")
    entryRange = [0,1,2,3,4]
    for i in entryRange:
        print(i, end="\t ")
        for j in range(3):
            print("%.2f" % d1_algoritmo[i][j], end=" - ")
        print("%.5f" % d1_algoritmo[i][3], end=" - ")
        print("\t",end="")
        for j in range(3):
            print("%.2f" % d2_algoritmo[i][j], end=" - ")
        print("%.5f" % d2_algoritmo[i][3], end=" - ")
        print("\t\t",end="")
        for j in range(3):
            print("%.2f" % d3_algoritmo[i][j],end=" - ")
        print("%.5f" % d3_algoritmo[i][3], end=" - ")
        print("\n",end="")
    
    print('Media', end="\t ")
    print_media(d1_algoritmo)
    print("\t",end="")
    print_media(d2_algoritmo)    
    print("\t\t",end="")
    print_media(d3_algoritmo)
    print("\n",end="")

    print('Std', end="\t ")
    print_desviacion(d1_algoritmo)
    print("\t",end="")
    print_desviacion(d2_algoritmo)    
    print("\t\t",end="")
    print_desviacion(d3_algoritmo)
    print("\n",end="")

# Loading datasets
ionosphere_features, ionosphere_target = load_arff(dataset_ionosphere)
parkinson_features, parkinson_target = load_arff(dataset_parkinson)
heart_features, heart_target = load_arff(dataset_heart)

# Normalizing data
ionosphere_features = MinMaxScaler().fit_transform(ionosphere_features)
parkinson_features = MinMaxScaler().fit_transform(parkinson_features)
ionosphere_features = MinMaxScaler().fit_transform(ionosphere_features)

# Obtaining results 
ionosphere_1NN, ionosphere_GR, ionosphere_LS = execution_results(dataset_ionosphere, ionosphere_features, ionosphere_target)
parkinson_1NN, parkinson_GR, parkinson_LS = execution_results(dataset_parkinson, parkinson_features, parkinson_target)
heart_spectf_1NN, heart_spectf_GR, heart_spectf_LS = execution_results(dataset_heart, heart_features, heart_target)

# Printing out
print("Finished algorithms\n")
print("\nResults for 1NN")
print_out(ionosphere_1NN, parkinson_1NN, heart_spectf_1NN)
print("\nResults for Greedy Relief")
print_out(ionosphere_GR, parkinson_GR, heart_spectf_GR)
print("\nResults for Local Search")
print_out(ionosphere_LS, parkinson_LS, heart_spectf_LS)