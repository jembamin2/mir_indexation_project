import numpy as np
import math
import cv2
#from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
import operator
import collections 
from collections import Counter


def euclidean(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    if l1.shape != l2.shape:
        print(f"Erreur : tailles différentes {l1.shape} vs {l2.shape}")
        return float('inf')  # ou raise une exception
    return np.linalg.norm(l1 - l2)

def manhattan(l1, l2):
    return np.sum(np.abs(np.array(l1) - np.array(l2)))

def chiSquareDistance(l1, l2):
    s = 0.0
    for i,j in zip(l1,l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
    return s

def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1,l2,dtype=np.float64)),dtype=np.float64)
    den = np.sqrt(np.sum(l1,dtype=np.float64)*np.sum(l2,dtype=np.float64))
    return math.sqrt( 1 - num / den )


def flann(a,b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)


def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName=="Manhattan":
        distance = manhattan(l1, l2)
    elif distanceName in ["Correlation","Chi carre","Intersection","Bhattacharyya"]:
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chi carre":
            distance = chiSquareDistance(l1, l2)
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Bhattacharyya":
            distance = bhatta(l1, l2)   
    elif distanceName=="Brute force":
        distance = bruteForceMatching(l1, l2)
    elif distanceName=="Flann":
        distance= flann(l1, l2)
    return distance

def getkVoisins(lfeatures, req, k,distanceName, combine=False): 

    #cas ou on a un seul descripteur
    if not combine:

        #on stock tout dans une liste de tuples (nom, feature, distance)
        ldistances = [] 
        for i in range(len(lfeatures)): 
            dist = distance_f(req, lfeatures[i][1],distanceName) #on calcule la distance entre la feature de l'image requete et la feature de l'image i
            ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
        if distanceName in ["Correlation","Intersection"]: #on inverse l'ordre si les distances sont meilleurs plus grandes
            ordre=True
        else:
            ordre=False
        ldistances.sort(key=operator.itemgetter(2),reverse=ordre) 

        #on récupère les k premiers éléments de la liste comme les k plus proches voisins
        lvoisins = [] 
        for i in range(k): 
            lvoisins.append(ldistances[i]) 
        return lvoisins
    
    else : #si on a plusieurs descripteurs
        ldistances = []

        #gérer les cas où il y a des features qui n'existent pas
        exist = [True for _ in range(4505)] 
        big_liste = []
        for i in lfeatures:
            liste = set()
            for j in i:
                value = j[0].split(".")[0]
                value = value.split("_")[-1]
                liste.add(int(value))
            big_liste.append(liste)
        
        for i in range(len(exist)):
            for j in big_liste:
                if i not in j:
                    exist[i] = False
                    break
        #on marque comme false quand sur une photo on a pas pu extraire une feature

        compteur = 0
        #on fait la même chose pour chaque descripteur
        for j in lfeatures:

            liste = []
            
            for i in range(len(j)):
                if not exist[i] and len(j) == 4505: #si on doit skip la feature
                    continue
                dist = distance_f(req[compteur], j[i][1], distanceName[compteur]) #calcul de la distance
                liste.append([j[i][0], j[i][1], dist])
            
            #si on doit inverser l'ordre des distances
            if distanceName[compteur] in ["Correlation", "Intersection"]:
                ordre = True
            else:
                ordre = False
            compteur += 1

            #normaliser les distances pour donner des poids équivalents à chaque descripteur
            max_dist = max(sublist[2] for sublist in liste)
            for sublist in liste:
                sublist[2] = sublist[2] / max_dist

            if ordre: #si on doit inverser l'ordre des distances alors on prend les opposés pour simplement inverser l'ordre
                for sublist in liste:
                    sublist[2] = 1 - sublist[2]

            ldistances.append(liste)

        #on additionne les distances de chaque descripteur pour chaque image
        trueldistances = []
        for i in range(len(ldistances[0])):
            distance = 0
            for j in range(len(ldistances)):
                distance += ldistances[j][i][2]
            trueldistances.append((ldistances[0][i][0], ldistances[0][i][1], distance))
        
        trueldistances.sort(key=operator.itemgetter(2))
        lvoisins = []   
        for i in range(k):
            lvoisins.append(trueldistances[i])
        return lvoisins


            