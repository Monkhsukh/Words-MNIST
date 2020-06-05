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

image_files = [f for f in sorted(glob.glob("dataset/dataset/v011_words_small/*.*"))]
with open('dataset/v011_labels_small.json') as json_file:
    data = json.load(json_file)

row_list = [["Path", "Actual word", "Kernel size", "Sharpened"]]
kernel_types = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]

for kernel_size in [3,5]:
	img_cnt = 1
	for x in image_files:
		img_name = x.split('/')[-1]
		img_cv = cv2.imread(x,0)
		img_cv = cv2.resize(img_cv,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

		kernel = np.ones((kernel_size, kernel_size), np.float32) / 9
		blurred = cv2.filter2D(img_cv, -1, kernel)
		kernel_sharpening = np.array([[-1,-1,-1], 
		                              [-1, 9,-1],
		                              [-1,-1,-1]])

		sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)

		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))
	 	cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
		
		dilation = cv2.dilate(sharpened,kernel,iterations = 1)

		print img_cnt, img_name
		print data[img_name], '=>',
		row = [img_name, data[img_name], str(kernel_size)]
		
		predicted_str = pytesseract.image_to_string(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
		row.append(predicted_str)
		cv2.imwrite('Outputs_sharpened/'+ 
			img_name.split('.')[0] + '_' + 
			str(kernel_size) + "." + 
			img_name.split('.')[1], sharpened) # save image file
		
		print row
		row_list.append(row)

		if img_cnt % 50 == 0 :
			print row_list
			#writeToFile(row_list)
			df = pd.DataFrame(data=row_list[1:],    # values
      			columns=row_list[0])
			row_list = []

			# if file does not exist write header 
			if not os.path.isfile('results_sharpened.csv'):
			   df.to_csv('results_sharpened.csv', index = False, header=True, encoding = 'utf-8')
			else: # else it exists so append without writing the header
			   df.to_csv('results_sharpened.csv', index = False, mode='a', header=True, encoding = 'utf-8')
			break
		img_cnt+=1
				
	
