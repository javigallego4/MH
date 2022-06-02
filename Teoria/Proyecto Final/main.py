import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from algorithms import *
from genetics import *
from memetics import *
from bbbc import *
from numpy.random import uniform
import time
import pandas as pd


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

"""Executes algorithms of P1 for given dataset.

    Returns:
        It returns results obtained for each of the algorithms         
        Results: Rows: partition-i. Cols: T-clas,T-red,Agr,T    

    Parameters:
    dataset -- dataset name
    dataset_features -- dataset features
    dataset target -- dataset target
    
    """
def execution_results_P1(dataset, dataset_features, dataset_target):    
    print('Algorithms belonging to P1')
    i = 0   # Indicates partition
    skf = StratifiedKFold(n_splits = 5) # cross-validation
    # Optimization
    entryRange_x = range(4)
    entryRange_y = range(7)
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

"""Executes algorithms of P2 for given dataset.

    Returns:
        It returns results obtained for each of the algorithms         
        Results: Rows: partition-i. Cols: T-clas,T-red,Agr,T    

    Parameters:
    dataset -- dataset name
    dataset_features -- dataset features
    dataset target -- dataset target
    
    """
def execution_results_P2(dataset, dataset_features, dataset_target):    
    print('Algorithms belonging to P2')
    "Indicates partition"
    i = 0   
    "Cross Validation"
    skf = StratifiedKFold(n_splits = 5) 
    "Optimization"
    entryRange_x = range(4)
    entryRange_y = range(7)    
    "AGG-BLX"
    dataset_AGGBLX = [[0 for x in entryRange_x] for y in entryRange_y]
    "AGE-BLX"
    dataset_AGEBLX = [[0 for x in entryRange_x] for y in entryRange_y]
    "AGG-AC"
    dataset_AGGAC = [[0 for x in entryRange_x] for y in entryRange_y]
    "AGE-AC"
    dataset_AGEAC = [[0 for x in entryRange_x] for y in entryRange_y]
    "AM(10,1.0)"
    dataset_AM1 = [[0 for x in entryRange_x] for y in entryRange_y]
    "AM(10,0.1)"
    dataset_AM2 = [[0 for x in entryRange_x] for y in entryRange_y]
    "AM(10,0.1*mej)"
    dataset_AM3 = [[0 for x in entryRange_x] for y in entryRange_y]

    "AGG-BLX convergence"
    convergence_AGGBLX = [[0] for y in range(5)]
    "AGG-AC convergence"
    convergence_AGGAC = [[0] for y in range(5)]
    "AGE-BLX convergence"
    convergence_AGEBLX = [[0] for y in range(5)]
    "AGE-AC convergence"
    convergence_AGEAC = [[0] for y in range(5)]
    convergence_AM1 = [[0] for y in range(5)]
    convergence_AM2 = [[0] for y in range(5)]
    convergence_AM3 = [[0] for y in range(5)]

    print("File: ", dataset)
    split_indexes = skf.split(dataset_features,dataset_target)
    for trainIndex , testIndex in split_indexes:
        n = testIndex.shape[0]                
        
        start = time.time()
        dataset_AGGBLX[i][0],dataset_AGGBLX[i][1], convergence_AGGBLX[i] = AGG(dataset_features,dataset_target,trainIndex,testIndex,BLX)
        dataset_AGGBLX[i][2] = 0.5*(dataset_AGGBLX[i][0] + dataset_AGGBLX[i][1])
        dataset_AGGBLX[i][3] = time.time()-start
        
        start = time.time()
        dataset_AGGAC[i][0],dataset_AGGAC[i][1], convergence_AGGAC[i] = AGG(dataset_features,dataset_target,trainIndex,testIndex,arithmeticCross)
        dataset_AGGAC[i][2] = 0.5*(dataset_AGGAC[i][0] + dataset_AGGAC[i][1])
        dataset_AGGAC[i][3] = time.time()-start

        start = time.time()
        dataset_AGEBLX[i][0],dataset_AGEBLX[i][1], convergence_AGEBLX[i] = AGE(dataset_features,dataset_target,trainIndex,testIndex,BLX)
        dataset_AGEBLX[i][2] = 0.5*(dataset_AGEBLX[i][0] + dataset_AGEBLX[i][1])
        dataset_AGEBLX[i][3] = time.time()-start
        
        start = time.time()
        dataset_AGEAC[i][0],dataset_AGEAC[i][1], convergence_AGEAC[i] = AGE(dataset_features,dataset_target,trainIndex,testIndex,arithmeticCross)
        dataset_AGEAC[i][2] = 0.5*(dataset_AGEAC[i][0] + dataset_AGEAC[i][1])
        dataset_AGEAC[i][3] = time.time()-start

        start = time.time()
        dataset_AM1[i][0],dataset_AM1[i][1], convergence_AM1[i] = AM(dataset_features,dataset_target,trainIndex,testIndex,BLX,am1)
        dataset_AM1[i][2] = 0.5*(dataset_AM1[i][0] + dataset_AM1[i][1])
        dataset_AM1[i][3] = time.time()-start
        
        start = time.time()
        dataset_AM2[i][0],dataset_AM2[i][1], convergence_AM2[i] = AM(dataset_features,dataset_target,trainIndex,testIndex,BLX,am2)
        dataset_AM2[i][2] = 0.5*(dataset_AM2[i][0] + dataset_AM2[i][1])
        dataset_AM2[i][3] = time.time()-start

        start = time.time()
        dataset_AM3[i][0],dataset_AM3[i][1], convergence_AM3[i] = AM(dataset_features,dataset_target,trainIndex,testIndex,BLX,am3)
        dataset_AM3[i][2] = 0.5*(dataset_AM3[i][0] + dataset_AM3[i][1])
        dataset_AM3[i][3] = time.time()-start
        
        i+=1
        print("Partition ",i)
    
    return dataset_AGGBLX, dataset_AGEBLX, dataset_AGGAC, dataset_AGEAC, dataset_AM1, dataset_AM2, dataset_AM3, convergence_AGGBLX, convergence_AGGAC, \
           convergence_AGEBLX, convergence_AGEAC, convergence_AM1, convergence_AM2, convergence_AM3
   
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
        d1_algoritmo[5][j] = media
        print("%.2f" % media, end=" & ")
    lista = []
    for i in range(5):
        lista.append(d1_algoritmo[i][3])
    media = np.mean(lista)
    d1_algoritmo[5][3] = media
    if len(str(media)) == 8:
        print("%.4f" % media, end="")
    else:
        print("%.5f" % media, end="")

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
        d_algoritmo[6][j] = desviacion
        print("%.2f" % desviacion, end=" & ")
    lista = []
    for i in range(5):
        lista.append(d_algoritmo[i][3])
    desviacion = np.std(lista)
    d_algoritmo[6][j] = desviacion
    if len(str(desviacion)) == 8:
        print("%.4f" % desviacion, end="")
    else:
        print("%.5f" % desviacion, end="")

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
        print(i, end="&\t ")
        for j in range(3):
            print("%.2f" % d1_algoritmo[i][j], end=" & ")
        print("%.5f" % d1_algoritmo[i][3], end=" & ")
        print("\t",end="")
        for j in range(3):
            print("%.2f" % d2_algoritmo[i][j], end=" & ")
        print("%.5f" % d2_algoritmo[i][3], end=" & ")
        print("\t\t",end="")
        for j in range(3):
            print("%.2f" % d3_algoritmo[i][j],end=" & ")
        print("%.5f" % d3_algoritmo[i][3], end=" \\ ")
        print("\n",end="")
    
    print('Media', end="&\t ")
    print_media(d1_algoritmo)
    print("\t",end="")
    print_media(d2_algoritmo)    
    print("\t\t",end="")
    print_media(d3_algoritmo)
    print("\n",end=" \\ ")

    print('Std', end="&\t ")
    print_desviacion(d1_algoritmo)
    print("\t",end="")
    print_desviacion(d2_algoritmo)    
    print("\t\t",end="")
    print_desviacion(d3_algoritmo)
    print("\n",end=" \\ ")

"""Creating CSV files of algorithms results for making charts.

    Parameters:
    arr -- array 2d 
    counter -- counter for saving name file

    """
def create_csv(arr, counter):
    dataset = pd.DataFrame({'T_Class': arr[:, 0], 'T_Red': arr[:, 1], 'Agr':arr[:,2], 'Tiempo':arr[:,3], 'Algoritmo':nombres[counter].split(sep='_')[1]})
    dataset.to_csv('Archivos_CSV/' + nombres[counter])

"Loading datasets"
ionosphere_features, ionosphere_target = load_arff(dataset_ionosphere)
parkinson_features, parkinson_target = load_arff(dataset_parkinson)
heart_features, heart_target = load_arff(dataset_heart)

"Normalizing data"
ionosphere_features = MinMaxScaler().fit_transform(ionosphere_features)
parkinson_features = MinMaxScaler().fit_transform(parkinson_features)
heart_features = MinMaxScaler().fit_transform(heart_features)

# "Obtaining results for algorithms belonging to P1"
# ionosphere_1NN, ionosphere_GR, ionosphere_LS = execution_results_P1(dataset_ionosphere, ionosphere_features, ionosphere_target)
# parkinson_1NN, parkinson_GR, parkinson_LS = execution_results_P1(dataset_parkinson, parkinson_features, parkinson_target)
# heart_spectf_1NN, heart_spectf_GR, heart_spectf_LS = execution_results_P1(dataset_heart, heart_features, heart_target)

# "Obtaining results for algorithms belonging to P2"
# ionosphere_AGGBLX, ionosphere_AGEBLX, ionosphere_AGGAC, \
# ionosphere_AGEAC, ionosphere_AM1, ionosphere_AM2, ionosphere_AM3, ionosphere_convergence_AGGBLX,\
# ionosphere_convergence_AGGAC, ionosphere_convergence_AGEBLX, ionosphere_convergence_AGEAC, \
# ionosphere_convergence_AM1, ionosphere_convergence_AM2, ionosphere_convergence_AM3 = execution_results_P2(dataset_ionosphere, ionosphere_features, ionosphere_target)

# parkinson_AGGBLX, parkinson_AGEBLX, parkinson_AGGAC, \
# parkinson_AGEAC, parkinson_AM1, parkinson_AM2, parkinson_AM3, parkinson_convergence_AGGBLX, \
# parkinson_convergence_AGGAC, parkinson_convergence_AGEBLX, parkinson_convergence_AGEAC, \
# parkinson_convergence_AM1, parkinson_convergence_AM2, parkinson_convergence_AM3 = execution_results_P2(dataset_parkinson, parkinson_features, parkinson_target)

# heart_spectf_AGGBLX, heart_spectf_AGEBLX, heart_spectf_AGGAC, \
# heart_spectf_AGEAC, heart_spectf_AM1, heart_spectf_AM2, heart_spectf_AM3, heart_convergence_AGGBLX, \
# heart_convergence_AGGAC, heart_convergence_AGEBLX, heart_convergence_AGEAC, \
# heart_convergence_AM1, heart_convergence_AM2, heart_convergence_AM3 = execution_results_P2(dataset_heart, heart_features, heart_target)

# # Printing out
# print("Finished algorithms\n")
# print("\nResults for 1NN")
# print_out(ionosphere_1NN, parkinson_1NN, heart_spectf_1NN)
# print("\nResults for Greedy Relief")
# print_out(ionosphere_GR, parkinson_GR, heart_spectf_GR)
# print("\nResults for Local Search")
# print_out(ionosphere_LS, parkinson_LS, heart_spectf_LS)
# print("\nResults for AGG-BLX")
# print_out(ionosphere_AGGBLX, parkinson_AGGBLX, heart_spectf_AGGBLX)
# print("\nResults for AGE-BLX")
# print_out(ionosphere_AGEBLX, parkinson_AGEBLX, heart_spectf_AGEBLX)
# print("\nResults for AGG-AC")
# print_out(ionosphere_AGGAC, parkinson_AGGAC, heart_spectf_AGGAC)
# print("\nResults for AGE-AC")
# print_out(ionosphere_AGEAC, parkinson_AGEAC, heart_spectf_AGEAC)
# print("\nResults for AM(10,1.0)")
# print_out(ionosphere_AM1, parkinson_AM1, heart_spectf_AM1)
# print("\nResults for AM(10,0.1)")
# print_out(ionosphere_AM2, parkinson_AM2, heart_spectf_AM2)
# print("\nResults for AM(10,0.1mej)")
# print_out(ionosphere_AM3, parkinson_AM3, heart_spectf_AM3)

# resultados = [ionosphere_1NN, parkinson_1NN, heart_spectf_1NN, ionosphere_GR, parkinson_GR, heart_spectf_GR, \
#     ionosphere_LS, parkinson_LS, heart_spectf_LS, ionosphere_AGGBLX, parkinson_AGGBLX, heart_spectf_AGGBLX, \
#     ionosphere_AGEBLX, parkinson_AGEBLX, heart_spectf_AGEBLX, ionosphere_AGGAC, parkinson_AGGAC, heart_spectf_AGGAC, \
#     ionosphere_AGEAC, parkinson_AGEAC, heart_spectf_AGEAC, ionosphere_AM1, parkinson_AM1, heart_spectf_AM1, \
#     ionosphere_AM2, parkinson_AM2, heart_spectf_AM2, ionosphere_AM3, parkinson_AM3, heart_spectf_AM3]

# nombres = ['ionosphere_1NN', 'parkinson_1NN', 'heartspectf_1NN', 'ionosphere_GR', 'parkinson_GR', 'heartspectf_GR', \
#     'ionosphere_LS', 'parkinson_LS', 'heartspectf_LS', 'ionosphere_AGGBLX', 'parkinson_AGGBLX', 'heartspectf_AGGBLX', \
#     'ionosphere_AGEBLX', 'parkinson_AGEBLX', 'heartspectf_AGEBLX', 'ionosphere_AGGAC', 'parkinson_AGGAC', 'heartspectf_AGGAC', \
#     'ionosphere_AGEAC', 'parkinson_AGEAC', 'heartspectf_AGEAC', 'ionosphere_AM1', 'parkinson_AM1', 'heartspectf_AM1', \
#     'ionosphere_AM2', 'parkinson_AM2', 'heartspectf_AM2', 'ionosphere_AM3', 'parkinson_AM3', 'heartspectf_AM3']

# 'Creating CSV files for making charts'
# print('Creando archivos .CSV')
# counter = 0
# for res in resultados:     
#     create_csv(np.array(res), counter)
#     counter += 1

# convergencias = [ionosphere_convergence_AGGBLX, ionosphere_convergence_AGGAC, ionosphere_convergence_AGEBLX, ionosphere_convergence_AGEAC, ionosphere_convergence_AM1, \
# ionosphere_convergence_AM2, ionosphere_convergence_AM3]

# nombres = ['ionosphere_convergence_AGGBLX', 'ionosphere_convergence_AGGAC', 'ionosphere_convergence_AGEBLX', 'ionosphere_convergence_AGEAC', \
# 'ionosphere_convergence_AM1', 'ionosphere_convergence_AM2', 'ionosphere_convergence_AM3']

# for i in range(len(convergencias)):
#     dataset = pd.DataFrame({'Agr': convergencias[i], 'Algoritmo':nombres[i].split(sep='_')[2]})
#     dataset.to_csv('Archivos_CSV/' + nombres[i])

def execute_BigBang(dataset, dataset_features, dataset_target):    
    print('Big Bang - Big Crunch Algorithm')
    "Indicates partition"
    i = 0   
    "Cross Validation"
    skf = StratifiedKFold(n_splits = 5) 
    "Optimization"
    entryRange_x = range(4)
    entryRange_y = range(7)    
    "BB-BC"
    dataset_BBBC = [[0 for x in entryRange_x] for y in entryRange_y]
    dataset_BBBC_LocalSearch = [[0 for x in entryRange_x] for y in entryRange_y]

    "AGG-BLX convergence"
    convergence_BBBC = [[0] for y in range(5)]
    convergence_BBBC_LocalSearch = [[0] for y in range(5)]

    print("File: ", dataset)
    split_indexes = skf.split(dataset_features,dataset_target)
    for trainIndex , testIndex in split_indexes:
        n = testIndex.shape[0]                
        
        start = time.time()
        dataset_BBBC[i][0],dataset_BBBC[i][1], convergence_BBBC = BBBC(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_BBBC[i][2] = 0.5*(dataset_BBBC[i][0] + dataset_BBBC[i][1])
        dataset_BBBC[i][3] = time.time()-start

        start = time.time()
        dataset_BBBC_LocalSearch[i][0],dataset_BBBC_LocalSearch[i][1], convergence_BBBC_LocalSearch = BBBC_LocalSearch(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_BBBC_LocalSearch[i][2] = 0.5*(dataset_BBBC_LocalSearch[i][0] + dataset_BBBC_LocalSearch[i][1])
        dataset_BBBC_LocalSearch[i][3] = time.time()-start
                
        i+=1
        print("Partition ",i)
    
    return dataset_BBBC, dataset_BBBC_LocalSearch, convergence_BBBC, convergence_BBBC_LocalSearch


ionosphere_BBBC, ionosphere_BBBC_LocalSearch, ionosphere_BBBC_convergence_, ionosphere_BBBC_LocalSearch_convergence = execute_BigBang(dataset_ionosphere, ionosphere_features, ionosphere_target)
parkinson_BBBC, parkinson_BBBC_LocalSearch, convergence_parkinson_BBBC, convergence_parkinson_BBBC_LocalSearch = execute_BigBang(dataset_parkinson, parkinson_features, parkinson_target)
heart_spectf_BBBC, heart_spectf_BBBC_LocalSearch, convergence_heart_BBBC, convergence_heart_BBBC_LocalSearch = execute_BigBang(dataset_heart, heart_features, heart_target)

# Printing out
print("Finished algorithm\n")
print("\nResults for BBBC")
print_out(ionosphere_BBBC, parkinson_BBBC, heart_spectf_BBBC)
print("\nResults for BBBC + Local Search")
print_out(ionosphere_BBBC_LocalSearch, parkinson_BBBC_LocalSearch, heart_spectf_BBBC_LocalSearch)

# resultados = [ionosphere_BBBC, parkinson_BBBC, heart_spectf_BBBC, ionosphere_BBBC_LocalSearch, parkinson_BBBC_LocalSearch, heart_spectf_BBBC_LocalSearch]
# nombres = ['ionosphere_BBBC', 'parkinson_BBBC', 'heartspectf_BBBC', 'ionosphere_BBBC+LocalSearch', 'parkinson_BBBC+LocalSearch', 'heartspectf_BBBC+LocalSearch']

# 'Creating CSV files for making charts'
# print('Creando archivos .CSV')
# counter = 0
# for res in resultados:     
#     create_csv(np.array(res), counter)
#     counter += 1

convergencias = [ionosphere_BBBC_convergence_, ionosphere_BBBC_LocalSearch_convergence]
nombres = ['ionosphere_BBBC_convergence', 'ionosphere_BBBC+LocalSearch_convergence']

for i in range(len(convergencias)):
    dataset = pd.DataFrame({'Agr': convergencias[i], 'Algoritmo':nombres[i].split(sep='_')[1]})
    dataset.to_csv('Archivos_CSV/' + nombres[i])