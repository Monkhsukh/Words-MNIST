try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import glob, os
import json
from pathlib import Path
import unicodecsv as csv
import cv2
import numpy as np
import pymorph
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg',0)
img_be = cv2.imread('0_v2.png',0)
img_the = cv2.imread('1.jpeg',0)

# So, I would start the procedure with these steps:
# -> sharpen operation (increase edges distance);
# -> Histogram equalization (because sharping will probably generate noise);
# -> Thresholding (Otsu's maybe, because equalization will not be enough to eliminate shadows);
# -> Closing operation (because thesholding will generate blank space, probably);

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

img = cv2.filter2D(img, -1, kernel_sharpening)
# img = cv2.equalizeHist(img)
            
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1) #
predicted_str = pytesseract.image_to_string(img)

print 'test'
print predicted_str

cv2.imshow('image',img)
cv2.waitKey(0)
# cv2.imshow('image',histeq)
# cv2.waitKey(0)
# cv2.imshow('image',threshold)
# cv2.waitKey(0)
# cv2.imshow('image',closing)
# cv2.waitKey(0)


cv2.destroyAllWindows()


# kernel_size = 3

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernel_size, kernel_size))

# # rows,cols = img.shape
# # M = np.float32([[1,0,2],[0,1,2]])
# # dst = cv2.warpAffine(img,M,(cols+6,rows+6), borderValue=(255,255,255))

# img = cv2.resize(img_or,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# # img = cv2.equalizeHist(img)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 2) #
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 2) #

# erode = cv2.erode(img,kernel, iterations = 1)
# dilate = cv2.dilate(img,kernel, iterations = 1)

# opening1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 1)
# closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1)

# plt.imshow(opening1)
# plt.show()

# plt.imshow(closing1)
# plt.show()

# plt.imshow(opening)
# plt.show()

# plt.imshow(closing)
# plt.show()

# plt.imshow(erode)
# plt.show()

# plt.imshow(dilate)
# plt.show()

# predicted_str = pytesseract.image_to_string(opening1)
# print '=',predicted_str
# predicted_str = pytesseract.image_to_string(closing1)
# print '=',predicted_str
# predicted_str = pytesseract.image_to_string(opening)
# print '=',predicted_str
# predicted_str = pytesseract.image_to_string(closing)
# print '=',predicted_str
# predicted_str = pytesseract.image_to_string(erode)
# print '=',predicted_str
# predicted_str = pytesseract.image_to_string(dilate)
# print '=',predicted_str

# kernel_3x3 = np.ones((3, 3), np.float32) / 9
# blurred = cv2.filter2D(img, -1, kernel_3x3)
# cv2.imshow('3x3 Kernel Blurring', blurred)
# cv2.waitKey(0)

# kernel_sharpening = np.array([[-1,-1,-1], 
#                               [-1, 9,-1],
#                               [-1,-1,-1]])

# sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)
# cv2.imshow('Image Sharpening', sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # predicted_str = pytesseract.image_to_string(sharpened)
# print '=',predicted_str

# img = cv2.equalizeHist(img)

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in xrange(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

# cv2.imshow('Image Sharpening', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# opening = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)
# erosion = cv2.erode(img_cv,kernel,iterations = 1)

# plt.figure()
# plt.imshow(img) 
# plt.show()