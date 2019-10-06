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
#fft得到的像谱默认不是按照中心对称的（快速傅里叶变换的原因），一般需要用fftshift方法使得其按中
# 心对称，这样的话当我们ifft 时，得到的数据就会和之前实际的不一样了，所以还需加ifftshift 来还原
dft_shift = np.fft.fftshift(dft)
idft      = np.fft.ifftshift(dft_shift)
dft_shift2 = np.fft.fftshift(idft)
#频谱，１３　相同
magnitude_spectrum1 = 20 * np.log(cv.magnitude(dft[:,:,0],            dft[:,:,1]))
magnitude_spectrum2 = 20 * np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
magnitude_spectrum3 = 20 * np.log(cv.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

# plt.subplot(141), plt.imshow(img,                cmap='gray')
# plt.subplot(142), plt.imshow(magnitude_spectrum1,cmap='gray')
# plt.subplot(143), plt.imshow(magnitude_spectrum2,cmap='gray')
# plt.subplot(144), plt.imshow(magnitude_spectrum3,cmap='gray')
# plt.show()

rows, cols = img.shape
print(img.shape)
crow, ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows,cols,2),np.uint8)
x=300
mask[crow-x:crow+x,ccol-x:ccol+x]= 1
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
# plt.subplot(121), plt.imshow(img,     cmap='gray')
# plt.subplot(122), plt.imshow(img_back,cmap='gray')
# plt.show()

#Different filter
# Why below arraies represent different filters?
mean_filter = np.ones((3,3))

x = cv.getGaussianKernel(5,10)
#T 对矩阵的一个转置，旋转９0度
gaussian = x * x.T

scharr = np.array([[ -3, 0, 3],
                   [-10, 0,10],
                   [ -3, 0, 3]])

sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_y = np.array([[-1,-2,-1],
                    [ 0, 0, 0],
                    [ 1, 2, 1]])

laplacian = np.array([[0, 1,0],
                      [1,-4,1],
                      [0, 1,0]])

filters = [mean_filter,gaussian,laplacian,sobel_x,sobel_y,scharr]
filters_name =['mean_filter','gaussian','laplacian','sobel_x','sobel_y','scharr']

fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    # print(magnitude_spectrum[i].shape)
    plt.subplot(2,3,i+1), plt.imshow(mag_spectrum[i],cmap='gray')
    plt.title(filters_name[i]),plt.xticks([]),plt.yticks([])
plt.show()