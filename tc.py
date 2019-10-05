import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
# For images, 2D Discrete Fourier Transform (DFT) is used to find the
# frequency domain. A fast algorithm called Fast Fourier Transform (FFT) is
# used for calculation of DFT.
# x(t)=Asin(2πft)
# Numpy has an FFT package to do this. np.fft.fft2()
img = cv.imread('/home/andy/me.jpeg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_spectrum = magnitude_spectrum/256  #why this step is need to imshow

# plt.subplot(121), plt.imshow(img,               cmap='gray')
# plt.subplot(122), plt.imshow(magnitude_spectrum,cmap='gray')
# plt.show()

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
xrang = 250
fshift[crow-xrang:crow+xrang, ccol-xrang:ccol+xrang] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
# plt.subplot(221), plt.imshow(img,cmap='gray'),      plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(img_back,cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(img_back),             plt.xticks([]), plt.yticks([])
# plt.show()

#cv method
# magnitude:震级；巨大；重大；重要性　spectrum:频谱；范围；领域；序列
dft = cv.dft(np.float32(img),flags= cv.DFT_COMPLEX_OUTPUT)
dft_shit = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shit[:,:,0],dft_shit[:,:,1]))
plt.subplot(121), plt.imshow(img,               cmap='gray')
plt.subplot(122), plt.imshow(magnitude_spectrum,cmap='gray')
plt.show()