import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from algorithms import *
from P3_algorithms import *
from numpy.random import uniform
import time
import pandas as pd
import pyinstrument as py
    


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
def execution_results_P3(dataset, dataset_features, dataset_target):    
    print('Algorithms belonging to P3')
    "Indicates partition"
    i = 0   
    "Cross Validation"
    skf = StratifiedKFold(n_splits = 5) 
    "Optimization"
    entryRange_x = range(4)
    entryRange_y = range(7)    
    "Basic Multi - Start search"
    dataset_BMB = [[0 for x in entryRange_x] for y in entryRange_y]
    "Simulated annealing"
    dataset_ES = [[0 for x in entryRange_x] for y in entryRange_y]
    "Iterative local search"
    dataset_ILS = [[0 for x in entryRange_x] for y in entryRange_y]
    "Hybrid between ILS - ES"
    dataset_hybrid = [[0 for x in entryRange_x] for y in entryRange_y]
    
    "BMB convergence"
    convergence_BMB = [[0] for y in range(5)]
    "ES convergence"
    convergence_ES = [[0] for y in range(5)]
    "ILS convergence"
    convergence_ILS = [[0] for y in range(5)]
    "ILS - ES convergence"
    convergence_HYBRID = [[0] for y in range(5)]

    print("File: ", dataset)
    split_indexes = skf.split(dataset_features,dataset_target)
    for trainIndex , testIndex in split_indexes:
        n = testIndex.shape[0]                
                
        print('BMB\n')

        start = time.time()
        dataset_BMB[i][0],dataset_BMB[i][1], convergence_BMB[i] = bmb(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_BMB[i][2] = 0.5*(dataset_BMB[i][0] + dataset_BMB[i][1])
        dataset_BMB[i][3] = time.time()-start

        print('ES\n')

        start = time.time()
        dataset_ES[i][0],dataset_ES[i][1], convergence_ES[i] = simulated_annealing(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_ES[i][2] = 0.5*(dataset_ES[i][0] + dataset_ES[i][1])
        dataset_ES[i][3] = time.time()-start

        print('ILS\n')

        start = time.time()
        dataset_ILS[i][0],dataset_ILS[i][1], convergence_ILS[i] = ils(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_ILS[i][2] = 0.5*(dataset_ILS[i][0] + dataset_ILS[i][1])
        dataset_ILS[i][3] = time.time()-start

        print('ILS-ES\n')

        start = time.time()
        dataset_hybrid[i][0],dataset_hybrid[i][1], convergence_HYBRID[i] = hybrid_ils_es(dataset_features,dataset_target,trainIndex,testIndex)
        dataset_hybrid[i][2] = 0.5*(dataset_hybrid[i][0] + dataset_hybrid[i][1])
        dataset_hybrid[i][3] = time.time()-start    

        i+=1
        print("Partition ",i)
    
    return dataset_BMB, dataset_ES, dataset_ILS, dataset_hybrid, convergence_BMB, convergence_ES, convergence_ILS, convergence_HYBRID
   
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
        



# ionosphere_1nn, ionosphere_gr, ionosphere_ls = execution_results_P1(dataset_ionosphere, ionosphere_features, ionosphere_target)
# ionosphere_1nn, ionosphere_gr, ionosphere_ls = execution_results_P1(dataset_parkinson, parkinson_features, parkinson_target)
# ionosphere_1nn, ionosphere_gr, ionosphere_ls = execution_results_P1(dataset_heart, heart_features, heart_target)



ionosphere_BMB, ionosphere_ES, ionosphere_ILS, ionosphere_HYBRID, \
ionosphere_convergence_BMB, ionosphere_convergence_ES, ionosphere_convergence_ILS, \
ionosphere_convergence_HYBRID = execution_results_P3(dataset_ionosphere, ionosphere_features, ionosphere_target)

parkinson_BMB, parkinson_ES, parkinson_ILS, parkinson_HYBRID, \
parkinson_convergence_BMB, parkinson_convergence_ES, parkinson_convergence_ILS, \
parkinson_convergence_HYBRID = execution_results_P3(dataset_parkinson, parkinson_features, parkinson_target)

heart_spectf_BMB, heart_spectf_ES, heart_spectf_ILS, heart_spectf_HYBRID, \
heart_spectf_convergence_BMB, heart_spectf_convergence_ES, heart_spectf_convergence_ILS, \
heart_spectf_convergence_HYBRID = execution_results_P3(dataset_heart, heart_features, heart_target)

# Printing out
print("Finished algorithms\n")
print("\nResults for BMB")
print_out(ionosphere_BMB, parkinson_BMB, heart_spectf_BMB)
print("\nResults for ES")
print_out(ionosphere_ES, parkinson_ES, heart_spectf_ES)
print("\nResults for ILS")
print_out(ionosphere_ILS, parkinson_ILS, heart_spectf_ILS)
print("\nResults for HYBRID ILS - ES")
print_out(ionosphere_HYBRID, parkinson_HYBRID, heart_spectf_HYBRID)

resultados = [ionosphere_BMB, ionosphere_ES, ionosphere_ILS, ionosphere_HYBRID, \
parkinson_BMB, parkinson_ES, parkinson_ILS, parkinson_HYBRID, \
heart_spectf_BMB, heart_spectf_ES, heart_spectf_ILS, heart_spectf_HYBRID]

nombres = ['ionosphere_BMB', 'ionosphere_ES', 'ionosphere_ILS', 'ionosphere_HYBRID', \
'parkinson_BMB', 'parkinson_ES', 'parkinson_ILS', 'parkinson_HYBRID', \
'heartspectf_BMB', 'heartspectf_ES', 'heartspectf_ILS', 'heartspectf_HYBRID']

'Creating CSV files for making charts'
print('Creando archivos .CSV')
counter = 0
for res in resultados:     
    create_csv(np.array(res), counter)
    counter += 1

convergencias = [ionosphere_convergence_BMB, ionosphere_convergence_ES, ionosphere_convergence_ILS, ionosphere_convergence_HYBRID]
# parkinson_convergence_BMB, parkinson_convergence_ES, parkinson_convergence_ILS, parkinson_convergence_HYBRID, 
# heart_spectf_convergence_BMB, heart_spectf_convergence_ES, heart_spectf_convergence_ILS, heart_spectf_convergence_HYBRID]

nombres = ['ionosphere_convergence_BMB', 'ionosphere_convergence_ES', 'ionosphere_convergence_ILS', 'ionosphere_convergence_HYBRID']

for i in range(len(convergencias)):
    dataset = pd.DataFrame({'Agr': convergencias[i], 'Algoritmo':nombres[i].split(sep='_')[2]})
    dataset.to_csv('Archivos_CSV/' + nombres[i])    
