#Defintion de toute les fonctions à appeller dans l'interface
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern

def showDialog():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    returnValue = msgBox.exec()

def generateHistogramme_Color(filenames, progressBar):
    if not os.path.isdir("BGR"):
        os.mkdir("BGR")
    i=0

    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

        num_image, _ = path.split(".")
        np.savetxt("BGR/"+str(num_image)+".txt" ,feature)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation Hist Couleur terminée !!!!")

def generateHistogramme_HSV(filenames, progressBar):
    if not os.path.isdir("HSV"):
        os.mkdir("HSV")
    i=0
    
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img],[0],None,[180],[0,180])
        histS = cv2.calcHist([img],[1],None,[256],[0,256])
        histV = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        num_image, _ = path.split(".")
        np.savetxt("HSV/"+str(num_image)+".txt" ,feature)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation Hist HSV terminée !!!!")
    
        
def generateSIFT(filenames, progressBar):
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        featureSum = 0
        sift = cv2.SIFT_create()  
        kps , des = sift.detectAndCompute(img,None)

        if des is None:
            print(path.split("/")[-1])
        else:

            num_image, _ = path.split(".")
            np.savetxt("SIFT/"+str(num_image)+".txt" ,des)
            progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        
            
            featureSum += len(kps)
        i+=1
    print("Indexation SIFT terminée !!!!")    


def generateORB(filenames, progressBar):
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        orb = cv2.ORB_create()
        key_point1,descrip1 = orb.detectAndCompute(img,None)
        
        if descrip1 is None:
            print(path)
        
        else : 
            num_image, _ = path.split(".")
            np.savetxt("ORB/"+str(num_image)+".txt" ,descrip1)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation ORB terminée !!!!")

def generateGLCM(filenames, progressBar):
    if not os.path.isdir("GLCM"):
        os.mkdir("GLCM")
    distances = [1, -1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    i = 0
    for path in os.listdir(filenames):
        image = cv2.imread(filenames+"/"+path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
        glcmProperties1 = greycoprops(glcmMatrix, 'contrast').ravel()
        glcmProperties2 = greycoprops(glcmMatrix, 'dissimilarity').ravel()
        glcmProperties3 = greycoprops(glcmMatrix, 'homogeneity').ravel()
        glcmProperties4 = greycoprops(glcmMatrix, 'energy').ravel()
        glcmProperties5 = greycoprops(glcmMatrix, 'correlation').ravel()
        glcmProperties6 = greycoprops(glcmMatrix, 'ASM').ravel()
        feature = np.array([glcmProperties1, glcmProperties2, glcmProperties3, glcmProperties4, glcmProperties5, glcmProperties6]).ravel()

        if feature is None:
            print(f"Nope {path}")
        else :
            image, _ = path.split(".")
            np.savetxt("GLCM/"+str(image)+".txt" ,feature)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i += 1
    print("indexation GLCM terminée !!!!")

def generateLBP(filenames, progressBar):
    if not os.path.isdir("LBP"):
        os.mkdir("LBP")
    i = 0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        points = 8
        radius = 1
        method = 'default'
        subSize = (70, 70)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (350, 350))
        fullLBPmatrix = local_binary_pattern(img, points, radius, method)
        histograms = []
        for k in range(int(fullLBPmatrix.shape[0] / subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1] / subSize[1])):
                subVector = fullLBPmatrix[k * subSize[0]:(k + 1) * subSize[0], j * subSize[1]:(j + 1) * subSize[1]].ravel()
                subHist, edges = np.histogram(subVector, bins=int(2 ** points), range=(0, 2 ** points))
                histograms = np.concatenate((histograms, subHist), axis=None)
            
        if histograms is None:
            print(f"Nope {path}")
        else :
            image, _ = path.split(".")
            np.savetxt("LBP/"+str(image)+".txt" ,histograms)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i += 1
    print("indexation LBP terminée !!!!")
