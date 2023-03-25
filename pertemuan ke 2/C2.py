import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUIP2.ui', self)
        self.Image = None
        self.button_LoadCitra.clicked.connect(self.fungsi)
        self.pushButton.clicked.connect(self.grayscale)

        # operasi titik
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContras_Streching.triggered.connect(self.stretching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)

        # operasi histogram
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)

        # operasi geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action_90_derajat.triggered.connect(self.rotasi90derajat)
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action_45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.actionskala_2x.triggered.connect(self.skala2x)
        self.actionskala_3x.triggered.connect(self.skala3x)
        self.actionskala_4x.triggered.connect(self.skala4x)
        self.actionskala1_2.triggered.connect(self.skalasetengah)
        self.actionskala1_4.triggered.connect(self.skalaseperempat)
        self.actionskala3_4X.triggered.connect(self.skalatigaperempat)
        self.actionCrop_Image.triggered.connect(self.crop)

        # operasi aritmatika
        self.actiontambah_dan_kurang.triggered.connect(self.tambahkurang)
        self.actionkali_dan_bagi.triggered.connect(self.kalidanbagi)
        self.actionoperasi_AND.triggered.connect(self.operasiAND)
        self.actionoperasi_OR.triggered.connect(self.operasiOr)
        self.actionoperasi_XOR.triggered.connect(self.operasiXor)


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

    def biner(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a < 180:
                    b = 1
                elif a > 180:
                    b = 255
                else:
                    b = 0

                self.Image.itemset((i, j), b)
        self.displayImage(1)

    def grayHistogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    def RGBHistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
        plt.plot(histo, color=col)
        plt.xlim([0, 256])
        plt.show()

    def EqualHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() -
                                               cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))

        self.Image = img
        self.displayImage(2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi90derajat(self):
        self.rotasi(-90)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasi45derajat(self):
        self.rotasi(-45)

    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(2)


    def skala2x(self):
        skala = 2
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 2X', resize_Image)
        cv2.waitKey()

    def skala3x(self):
        skala = 3
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 3X', resize_Image)
        cv2.waitKey()

    def skala4x(self):
        skala = 4
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 4X', resize_Image)
        cv2.waitKey()


    def skalasetengah(self):
        skala = 0.5
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 1/2', resize_Image)
        cv2.waitKey()

    def skalaseperempat(self):
        skala = 0.25
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 1/4', resize_Image)
        cv2.waitKey()

    def skalatigaperempat(self):
        skala = 0.75
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in 3/4', resize_Image)
        cv2.waitKey()

    def crop(self):
        img = cv2.imread('paru-paru2.png')
        cut_img = img[0: 500, 0: 500]
        cv2.imshow("cut_img", cut_img)

    def tambahkurang(self):
        gambar1 = cv2.imread('paru-paru2.png', 0)
        gambar2 = cv2.imread('paru-paru2.png', 0)
        test1 = gambar1 + gambar2
        test2 = gambar1 - gambar2
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image 1 tambah', test1)
        cv2.imshow('Image 1 kurang', test2)
        cv2.waitKey()

    def kalidanbagi(self):
        gambar1 = cv2.imread('paru-paru2.png', 0)
        gambar2 = cv2.imread('paru-paru2.png', 0)
        test1 = gambar1 * gambar2
        test2 = gambar1 / gambar2
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image 1 kali', test1)
        cv2.imshow('Image 1 bagi', test2)
        cv2.waitKey()

    def operasiAND(self):
        gambar1 = cv2.imread('paru-paru2.png', 1)
        gambar2 = cv2.imread('paru-paru2.png', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi AND', operasi)
        cv2.waitKey()

    def operasiOr(self):
        gambar1 = cv2.imread('paru-paru2.png', 1)
        gambar2 = cv2.imread('paru-paru2.png', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi and', operasi)
        cv2.waitKey()

    def operasiXor(self):
        gambar1 = cv2.imread('paru-paru2.png', 1)
        gambar2 = cv2.imread('paru-paru2.png', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi and', operasi)
        cv2.waitKey()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
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