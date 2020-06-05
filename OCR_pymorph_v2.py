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

image_files = [f for f in sorted(glob.glob("dataset/dataset/v011_words_small/*.*"))]

row_list = [["Path", "Actual word", "Pymorph"]]
cnt = 0	
kernel = np.ones((5,5),np.uint8)

img_cv = cv2.imread('1005.jpeg',0)
opening = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)
erosion = cv2.erode(img_cv,kernel,iterations = 1)

cv2.imshow('image',img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.figure()
# plt.imshow(closing) 
# plt.show()
# plt.figure()
# plt.imshow(erosion) 
# plt.show()

#a=readgray('100.png')
# c = Image.open('100.png')
# color_img = np.asarray(Image.open('100.png')) / 255
# gray_img = rgb2gray(color_img)

# a = Image.open('100.png').convert('LA')
# arr = np.array(a.getdata(), dtype=np.uint8)


# print gray_img

# plt.show()

#plt.imshow(arr, interpolation='nearest')
#plt.show()

# plt.figure()
# plt.imshow(arr) 
# plt.show()


# for x in image_files:
# 	img_name = x.split('/')[-1]
   
# 	prediction = pytesseract.image_to_string(img_cv)

# 	cnt+=1

# 	print cnt, img_name
# 	print data[img_name], '=>',prediction
	
# 	row_list.append([img_name, data[img_name],prediction])
# 	if cnt % 50 == 0 :
#     		print row_list
#     		writeToFile(row_list)
#     		row_list = []
