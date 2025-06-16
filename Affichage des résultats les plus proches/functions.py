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

def generateSIFT(filenames, progressBar):
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        featureSum = 0
        sift = cv2.SIFT_create()  
        kps , des = sift.detectAndCompute(img,None)

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
        
        num_image, _ = path.split(".")
        np.savetxt("ORB/"+str(num_image)+".txt" ,descrip1 )
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation ORB terminée !!!!")
	
def extractReqFeatures(fileName,algo_choice):  
    print(algo_choice)
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif algo_choice==3: #SIFT
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif algo_choice==4: #ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)

        elif algo_choice==5: #GLCM
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = img_as_ubyte(img_gray)
            '''
            glcm = greycomatrix(img_gray, [1], [0], 256, symmetric=True, normed=True)
            vect_features = np.array([greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0]])
            '''
            distances = [1, -1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = greycomatrix(img_gray, distances=distances, angles=angles, normed=True, symmetric=True)

            vect_features = np.array([
                greycoprops(glcm, 'contrast').ravel(),
                greycoprops(glcm, 'dissimilarity').ravel(),
                greycoprops(glcm, 'homogeneity').ravel(),
                greycoprops(glcm, 'energy').ravel(),
                greycoprops(glcm, 'correlation').ravel(),
                greycoprops(glcm, 'ASM').ravel()
            ]).ravel()
        elif algo_choice==6: #LBP
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
            vect_features = hist
			
        np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        print("saved")
        #print("vect_features", vect_features)
        return vect_features