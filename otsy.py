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
from collections import OrderedDict
import pandas as pd

row_list = [["Path", "Actual word", 'Result']]

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

image_files = [f for f in sorted(glob.glob("dataset/dataset/v011_words_small/*.*"))]
with open('dataset/v011_labels_small.json') as json_file:
    data = json.load(json_file)

img_cnt = 1
for x in image_files:
	print img_cnt
	img_name = x.split('/')[-1]
	img_cv = cv2.imread(x,0)
	
	morph_result = OrderedDict()
	
	kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

	img = cv2.filter2D(img_cv, -1, kernel_sharpening)
	# img = cv2.equalizeHist(img)
	            
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1) #

	# img = cv2.filter2D(img_cv, -1, kernel_sharpening)
	# #img = cv2.equalizeHist(img)
	            
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
	# img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1) #
	# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 1) #
	predicted_str = pytesseract.image_to_string(img)


	print img_cnt, img_name
	print data[img_name], '=>',
	row = [img_name, data[img_name], predicted_str]
	
	# cv2.imwrite('Outputs_otsu/'+ 
	# 	img_name.split('.')[0] + '_' +
	# 	"." +  
	# 	img_name.split('.')[1], img) # save image file
	

	print row
	row_list.append(row)

	if img_cnt % 50 == 0 :
		print row_list
		df = pd.DataFrame(data=row_list[1:],    # values
  			columns=row_list[0])
		row_list = []

		# if file does not exist write header 
		if not os.path.isfile('results_otsu.csv'):
		   df.to_csv('results_otsu.csv', index = False, header=True, encoding = 'utf-8')
		else: # else it exists so append without writing the header
		   df.to_csv('results_otsu.csv', index = False, mode='a', header=True, encoding = 'utf-8')
	img_cnt+=1
				
	
