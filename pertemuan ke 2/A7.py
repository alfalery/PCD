import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUIP2.ui', self)
        self.Image = None
        self.button_LoadCitra.clicked.connect(self.fungsi)
        self.pushButton.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContras_Streching.triggered.connect(self.stretching)
        self.actionNegative.triggered.connect(self.negative)

    def fungsi(self):
        self.Image = cv2.imread('paru-paru2.png')
        self.displayImage()

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)


        self.displayImage(1)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(1)

    def stretching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(1)

    def negative(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        negative = 255

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = negative - a

                self.Image.itemset((i, j), b)
        self.displayImage(1)

    def displayImage(self,windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('p1')
window.show()
sys.exit(app.exec_())