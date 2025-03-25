# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TP2_indexation.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
import functions
import time 

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1038, 595)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 120, 441, 31))
        self.label.setFrameShape(QtWidgets.QFrame.Panel)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.imagelabel = QtWidgets.QLabel(self.centralwidget)
        self.imagelabel.setGeometry(QtCore.QRect(10, 110, 261, 231))
        self.imagelabel.setText("")
        self.imagelabel.setObjectName("imagelabel")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(10, 160, 441, 331))
        self.image.setFrameShape(QtWidgets.QFrame.Panel)
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 120, 531, 31))
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.charger = QtWidgets.QPushButton(self.centralwidget)
        self.charger.setGeometry(QtCore.QRect(10, 50, 441, 51))
        self.charger.setObjectName("charger")
        self.indexer = QtWidgets.QPushButton(self.centralwidget)
        self.indexer.setGeometry(QtCore.QRect(470, 50, 451, 51))
        self.indexer.setObjectName("indexer")
        self.checkBox_SIFT = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_SIFT.setGeometry(QtCore.QRect(20, 10, 85, 21))
        self.checkBox_SIFT.setObjectName("checkBox_SIFT")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(470, 160, 531, 331))
        self.tableView.setObjectName("tableView")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 510, 1011, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.checkBox_ORB = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_ORB.setGeometry(QtCore.QRect(140, 10, 85, 21))
        self.checkBox_ORB.setObjectName("checkBox_ORB")
        self.checkBox_HistC = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HistC.setGeometry(QtCore.QRect(270, 10, 121, 21))
        self.checkBox_HistC.setObjectName("checkBox_HistC")
        self.checkBox_HSV = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HSV.setGeometry(QtCore.QRect(470, 10, 101, 21))
        self.checkBox_HSV.setObjectName("checkBox_HSV")
        self.checkBox_GLCM = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_GLCM.setGeometry(QtCore.QRect(650, 10, 81, 21))
        self.checkBox_GLCM.setObjectName("checkBox_GLCM")
        self.indexer_2 = QtWidgets.QPushButton(self.centralwidget)
        self.indexer_2.setGeometry(QtCore.QRect(930, 50, 71, 51))
        self.indexer_2.setObjectName("indexer_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1038, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.charger.clicked.connect(self.Ouvrir)
        self.tableView.clicked.connect(self.cliquerTab)
        self.indexer_2.clicked.connect(self.Exit)
        self.indexer.clicked.connect(self.extractFeatures)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Image"))
        self.label_3.setText(_translate("MainWindow", "Base d\'images"))
        self.charger.setText(_translate("MainWindow", "Charger et afficher la base de données"))
        self.indexer.setText(_translate("MainWindow", "Calculer les descripteurs"))
        self.checkBox_SIFT.setText(_translate("MainWindow", "SIFT"))
        self.checkBox_ORB.setText(_translate("MainWindow", "ORB"))
        self.checkBox_HistC.setText(_translate("MainWindow", "Hist Couleur"))
        self.checkBox_HSV.setText(_translate("MainWindow", "Hist HSV"))
        self.checkBox_GLCM.setText(_translate("MainWindow", "GLCM"))
        self.indexer_2.setText(_translate("MainWindow", "Quitter"))
    
    def Ouvrir(self, MainWindow):
        self.list_images = []

        # Sélectionner le dossier contenant les images
        self.Dossier_images = QtWidgets.QFileDialog.getExistingDirectory(
            None, 'Select directory', "C://", QtWidgets.QFileDialog.ShowDirsOnly
        ) + "/"

        # Récupérer tout dans les sous dossiers
        for root, _, files in os.walk(self.Dossier_images):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Filtrer les images
                    self.list_images.append(os.path.join(root, file))
        print(self.list_images[0])
        # affichage de la première image
        if self.list_images:
            pixmap = QtGui.QPixmap(self.list_images[0])
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.image.setAlignment(QtCore.Qt.AlignCenter)

        # Remplir le modèle de données pour la tableView
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["File Name"])

        for image_path in self.list_images:
            item = QStandardItem(image_path)
            item.setEditable(False)
            model.appendRow([item])

        self.tableView.setModel(model)
    
    def cliquerTab (self, MainWindow):
        index = self.tableView.selectionModel().currentIndex()
        UrlImg=index.sibling(index.row(),index.column()).data()
        pixmap = QtGui.QPixmap(UrlImg)
        pixmap = pixmap.scaled(self.image.width(), self.image.height(),QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)
        self.image.setAlignment(QtCore.Qt.AlignCenter)

    def Exit(self, MainWindow):
        sys.exit()

    def extractFeatures(self,MainWindow):  
        #Calculer les descripteurs
        if self.Dossier_images and self.checkBox_SIFT.isChecked():
            start = time.time()
            functions.generateSIFT(self.Dossier_images, self.progressBar,self.list_images)
            print(time.time()-start)

        if self.Dossier_images and self.checkBox_ORB.isChecked():
            start = time.time()
            functions.generateORB(self.Dossier_images, self.progressBar,self.list_images)
            print(time.time()-start)

        if self.Dossier_images and self.checkBox_HistC.isChecked():
            start = time.time()
            functions.generateHistogramme_Color(self.Dossier_images, self.progressBar,self.list_images)
            print(time.time()-start)

        if self.Dossier_images and self.checkBox_HSV.isChecked():
            start = time.time()
            functions.generateHistogramme_HSV(self.Dossier_images, self.progressBar,self.list_images)
            print(time.time()-start)

        if self.Dossier_images and self.checkBox_GLCM.isChecked():
            start = time.time()
            functions.generateGLCM(self.Dossier_images, self.progressBar)
            print(time.time()-start)

        if not self.checkBox_SIFT.isChecked() and not self.checkBox_HistC.isChecked() and not self.checkBox_HSV.isChecked() and not self.checkBox_ORB.isChecked() and not self.checkBox_GLCM.isChecked() and not self.checkBox_LBP.isChecked():
            print("Merci de selectionner un descripteur via le Menu ...")
            functions.showDialog()
        if len(self.Dossier_images)<1:
            print("Merci de charger la base de données avec le bouton Ouvrir")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
