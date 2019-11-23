import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

# readFile: function for read images from directory in the project
# returns list of the file path of each image going to be read
def readFile(dir):

    # list the file of
    files = os.listdir(dir)

    filePath = []
    for file in files:
        completePath = dir + file
        filePath.append(completePath)

    return filePath

# checkSize: function for check size of the source image (DIP book images)
# and destination image (images in "Origin Image" directory)
def checkSize(src, dst):

    # get size of source image
    xSrc, ySrc = src.shape

    # get size of destination image
    xDst, yDst = dst.shape

    # If the size of destination image is not the same as source image,
    # the destination image will be resized into the same size as source image
    # Otherwise, the size of destination image will remain the same.
    if xSrc != xDst and ySrc != yDst:
        height, width = src.shape[:2]
        resize = cv2.resize(dst, (width, height), interpolation=cv2.INTER_CUBIC)
        return resize
    else:
        return dst

# dft_shift: DFT Shift function, return shifted frequency in horizontal & vertical direction
def dft_shift(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    return f_shift

# idft_shift: IDFT Shift function, returns the image in spatial format
def ifft_shitf(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift)

    return img_back

# normalize function: return the integer 8bit image (0-255) of denoised image
def normalize(img):
    img = np.abs(img)
    img -= img.min()
    img *= 255.0/img.max()
    return img.astype(np.uint8)

# Blur2Filter: function for apply blur2filter image
def applyBlur2Filter(img, xSize, ySize, func):
    image = np.copy(img)

    # Construct image from blurring function
    for u in range(0, xSize):
        for v in range(0, ySize):
            image[u,v] = func(u - xSize/2, v - ySize/2)

    # Do the blurring by multiply normal image and blur image
    return image*img

# blurring function H(u,v) = (T / pi(ua + vb))sin(pi(ua + vb))exp(-jpi(ua + vb))
def blur2Filter(x,y):
    a = 0.1
    b = 0.1
    T = 1
    C = math.pi*(a*x+b*y)

    if(C == 0):
        return 1

    return (T/C)*math.sin(C)*math.e**(-1j*C)

# blur1: generate blur image by using estimated function H(u,v) = G(u,v) / F(u,v) in frequency domain
# G(u,v) = blurred DIP cover
# F(u,v) = original DIP cover
def blur1(blur, ori, img):
    imgProcess = checkSize(DIP_blur, imgBegin)

    # do the DFT on each image
    G = dft_shift(DIP_blur)
    F = dft_shift(DIP_ori)
    imgFFT = dft_shift(imgProcess)

    # calculate the blurring function by dividing Blurred DIP book Cover with Original DIP Book Cover
    # in frequency domain
    H = G / F

    # process the images from "Origin Image" directory with
    img_H_filter = H * imgFFT
    imgResult = ifft_shitf(img_H_filter)

    # remove the noise on filtered image in spatial domain
    denoise = normalize(imgResult)

    imgEnd = checkSize(imgBegin, denoise)

    return imgEnd

# blur2: geberate blur image by using modelling function
# H(u,v) = (T / pi(ua + vb))sin(pi(ua + vb))exp(-jpi(ua + vb))
def blur2(img):
    # do DFT on image going to be blurred
    imgFFT = dft_shift(img)

    # get the size of source image
    xSize, ySize = imgFFT.shape

    # apply blurring filter based on modelled degradation function in frequency domain
    img_H_filter = applyBlur2Filter(imgFFT, xSize, ySize, blur2Filter)

    # do IDFT on filtered image to return in
    imgResult = ifft_shitf(img_H_filter)

    # Remove the noise on the image
    denoise = normalize(imgResult)

    return denoise


# image folder path
oriImg = "Origin Image/"
DIP = "DIPBook/"

# read each file on the folder
pathOriImg = readFile(oriImg)
pathDIP = readFile(DIP)

# Load DIP image in grayscale
DIP_blur = cv2.imread(pathDIP[0], 0)
DIP_ori = cv2.imread(pathDIP[1], 0)

# If "Image Result" Directory doesn't exist, create the directory named "Image Result"
if not os.path.exists("Image Result"):
    os.mkdir("Image Result")

# generate blurred image in folder "Origin Images"
for i in range(0, len(pathOriImg)):
    print("Processing Image: Q" + str(i+1))

    # Load "origin image" in grayscale
    imgBegin = cv2.imread(pathOriImg[i], 0)

    # perform blurring function
    imgBlur1 = blur1(DIP_blur, DIP_ori, imgBegin)
    imgBlur2 = blur2(imgBegin)

    # Plot the difference between estimation by observation (problem A) & estimation by modelling (problem B)
    plt.subplot(231), plt.imshow(imgBegin, cmap = 'gray')
    plt.title('Origin Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(imgBlur1, cmap='gray')
    plt.title('Problem A'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(imgBlur2, cmap='gray')
    plt.title('Problem B'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.hist(imgBegin.ravel(), 256, [0,256])
    plt.title('Origin Image Hist'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.hist(imgBlur1.ravel(), 256, [0, 256])
    plt.title('Problem A Hist'), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.hist(imgBlur2.ravel(), 256, [0, 256])
    plt.title('Problem B Hist'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Save Generated image on "Image Result" directory
    name = "Image Result/Q" + str(i+1)
    cv2.imwrite(name + " Problem A.jpg", imgBlur1)
    cv2.imwrite(name + " Problem B.jpg", imgBlur2)