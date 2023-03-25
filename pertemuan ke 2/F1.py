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
        loadUi('GUI.ui', self)
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
        self.actionCrop_Image.triggered.connect(self.crop)

        # operasi aritmatika
        self.actiontambah_dan_kurang.triggered.connect(self.tambahkurang)
        self.actionoperasi_AND.triggered.connect(self.operasiAND)
        self.actionoperasi_OR.triggered.connect(self.operasiOr)
        self.actionoperasi_XOR.triggered.connect(self.operasiXor)
        # Operasi Spasial

        self.actionMean.triggered.connect(self.mean)
        self.actionMedian.triggered.connect(self.median)
        self.actionMaxfilter.triggered.connect(self.maxfilter)
        self.actionMinfilter.triggered.connect(self.MinFilter)

        #Operasi Transform
        self.actionDFT_Smoothing_Image_2.triggered.connect(self.smoothimage)
        self.actionDFT_Smoothing_Image_Tepi_2.triggered.connect(self.DFTtepi)
        self.actionDeteksi_Tepi_2.triggered.connect(self.Sobel)
        self.actionPrewitt_2.triggered.connect(self.Prewitt)
        self.actionRobert_2.triggered.connect(self.robert)

    def fungsi(self):
        self.Image = cv2.imread('noise.jpg')
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

    def zoomin(self):
        skala = 2
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in', resize_Image)
        cv2.waitKey()

    def zoomout(self):
        skala = 0.5
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom in', resize_Image)
        cv2.waitKey()

    def crop(self):
        img = cv2.imread('pegunungan.jpg')
        cut_img = img[0: 500, 0: 500]
        cv2.imshow("cut_img", cut_img)

    def tambahkurang(self):
        gambar1 = cv2.imread('pegunungan.jpg', 0)
        gambar2 = cv2.imread('padi.jpg', 0)
        test1 = gambar1 + gambar2
        test2 = gambar1 - gambar2
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image 1 tambah', test1)
        cv2.imshow('Image 1 kurang', test2)
        cv2.waitKey()

    def operasiAND(self):
        gambar1 = cv2.imread('pegunungan.jpg', 1)
        gambar2 = cv2.imread('padi.jpg', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi AND', operasi)
        cv2.waitKey()

    def operasiOr(self):
        gambar1 = cv2.imread('pegunungan.jpg', 1)
        gambar2 = cv2.imread('padi.jpg', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi and', operasi)
        cv2.waitKey()

    def operasiXor(self):
        gambar1 = cv2.imread('pegunungan.jpg', 1)
        gambar2 = cv2.imread('padi.jpg', 1)
        gambar1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
        gambar2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(gambar1, gambar2)
        cv2.imshow('Image 1 original', gambar1)
        cv2.imshow('Image 2 original', gambar2)
        cv2.imshow('Image Operasi and', operasi)
        cv2.waitKey()

    def conv(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]
        F_height = F.shape[0]
        F_width = F.shape[1]
        H = (F_height) // 2
        W = (F_width) // 2
        out = np.zeros((X_height, X_width))
        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def conv2(self, X, F):
        X_Height = X.shape[0]
        X_Width = X.shape[1]

        F_Height = F.shape[0]
        F_Width = F.shape[1]

        H = 0
        W = 0

        batas = (F_Height) // 2

        out = np.zeros((X_Height, X_Width))

        for i in np.arange(H, X_Height - batas):
            for j in np.arange(W, X_Width - batas):
                sum = 0
                for k in np.arange(H, F_Height):
                    for l in np.arange(W, F_Width):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def konvolusi2d(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def mean(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 9) * np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def imgsharp(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def median(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                hasil.itemset((i, j), b)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def maxfilter(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                hasil.itemset((i, j), b)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def MinFilter(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                    b = min
                    img_out.itemset((i, j), b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def smoothimage(self):
        x=np.arange(256)
        y=np.sin(2*np.pi*x/3)

        y+=max(y)

        Img=np.array([[y[j]*127 for j in range (256)]for i in
        range(256)],dtype=np.uint8)

        plt.imshow(Img)
        Img=cv2.imread('noise.png',0)

        dft=cv2.dft(np.float32(Img),flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift=np.fft.fftshift(dft)

        magnitude_spectrum=20*np.log((cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
        rows,cols = Img.shape
        crow, ccol = int(rows/2),int(cols/2)
        mask = np.zeros((rows,cols,2),np.uint8)
        r = 50
        center = [crow,ccol]
        x,y = np.ogrid[:rows, :cols]
        mask_area = (x - center [0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1

        fshift=dft_shift*mask
        fshift_mask_mag=20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        f_shift=np.fft.ifftshift(fshift)

        Img_back=cv2.idft(f_shift)
        Img_back=cv2.magnitude(Img_back[:,:,0],Img_back[:,:,1])

        fig=plt.figure(figsize=(12,12))
        ax1=fig.add_subplot(2,2,1)
        ax1.imshow(Img, cmap='gray')
        ax1.title.set_text('input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(Img_back, cmap='gray')
        ax4.title.set_text('Inverse fourier')
        plt.show()

    def DFTtepi(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        plt.imshow(img)
        img = cv2.imread('noise.png', 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def Sobel(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Prewitt(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        self.Image = img
        self.displayImage(2)
        print(img)
        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def robert(self):
        path, _ = QFileDialog.getOpenFileName()
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[1, 0],
                       [0, -1]])
        Sy = np.array([[0, 1],
                       [-1, 0]])
        img_x = self.conv2(img, Sx)
        img_y = self.conv2(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out))* 255
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()



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