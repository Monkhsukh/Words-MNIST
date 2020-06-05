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

def writeToFile(row_list):
    file_name = 'results_resized.csv'
    if Path(file_name).is_file():
        f = open(file_name, "a")
        sniffer = csv.Sniffer()
        csv_dialect = sniffer.sniff(
            open(file_name).readline())

        writer = csv.writer(f, encoding='UTF-8', quoting = csv.QUOTE_NONE, escapechar='|', dialect=csv_dialect)
        writer.writerows(row_list)
        f.close()
    else:
        with open(file_name,'w') as file:
            writer = csv.writer(file,encoding='UTF-8', delimiter=',', quoting=csv.QUOTE_NONE, escapechar='|')
            writer.writerows(row_list)

image_files = [f for f in sorted(glob.glob("dataset/dataset/v011_words_small/*.*"))]
with open('dataset/v011_labels_small.json') as json_file:
    data = json.load(json_file)

row_list = [["Path", "Actual word", "Kernel size", "Kernel type", "Iterations", "Original", "Opening", "Closing", "Erosion", "Dilation", "OC", "CO","OCO", "COC", 'Sharpened']]
kernel_types = [cv2.MORPH_CROSS] #MORPH_RECT, cv2.MORPH_CROSS] #, cv2.MORPH_ELLIPSE]

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
for number_of_iteration in [1]:
	for kernel_size in [3]:
		for kernel_type in kernel_types:
			img_cnt = 1
			for x in image_files:
				print img_cnt
				img_name = x.split('/')[-1]
				img_cv = cv2.imread(x,0)
				img_cv = cv2.resize(img_cv,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
				
				# rows,cols = img_cv.shape
				# M = np.float32([[1,0,2],[0,1,2]])
				# img_cv = cv2.warpAffine(img_cv,M,(cols+6,rows+6), borderValue=(255,255,255))
				
				kernel = cv2.getStructuringElement(kernel_type,(kernel_size, kernel_size))

				morph_result = OrderedDict()
				#morph_result['original'] = img_cv
				morph_result.update({'opening' : cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel, iterations = number_of_iteration)}) # opening
				morph_result.update({'closing' : cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel, iterations = number_of_iteration)}) # closing
				morph_result.update({'erode' : cv2.erode(img_cv,kernel, iterations = number_of_iteration)}) # erode
				morph_result.update({'dilate' : cv2.dilate(img_cv,kernel, iterations = number_of_iteration)}) # dilate
				morph_result.update({'OC' : cv2.morphologyEx(morph_result['opening'], cv2.MORPH_CLOSE, kernel, iterations = number_of_iteration)}) # OC
				morph_result.update({'CO' : cv2.morphologyEx(morph_result['closing'], cv2.MORPH_OPEN, kernel, iterations = number_of_iteration)}) # CO
				morph_result.update({'OCO' : cv2.morphologyEx(morph_result['OC'], cv2.MORPH_OPEN, kernel, iterations = number_of_iteration)}) # OCO
				morph_result.update({'COC' : cv2.morphologyEx(morph_result['CO'], cv2.MORPH_CLOSE, kernel, iterations = number_of_iteration)}) # COC

				kernel_blur = np.ones((kernel_size*2-1, kernel_size*2-1), np.float32) / 9
				blurred = cv2.filter2D(img_cv, -1, kernel_blur)
				sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)
				morph_result.update({'Sharpened' : sharpened})


				print img_cnt, img_name
				print data[img_name], '=>',
				row = [img_name, data[img_name], str(kernel_size), str(kernel_type), str(number_of_iteration),'']
				sub_cnt = 1
				for (key, img) in morph_result.items():
					predicted_str = pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
					row.append(predicted_str)
					
					if img_cnt < 10 :
						cv2.imwrite('Outputs_final/'+ 
							img_name.split('.')[0] + '_' + 
							str(number_of_iteration) + 
							str(kernel_size)+ 
							str(kernel_type) + '_' + 
							key + "." + 
							img_name.split('.')[1], img) # save image file
					
					sub_cnt+=1

				print row
				row_list.append(row)

				if img_cnt % 50 == 0 :
					print row_list
					#writeToFile(row_list)
					df = pd.DataFrame(data=row_list[1:],    # values
              			columns=row_list[0])
					row_list = []

					# if file does not exist write header 
					if not os.path.isfile('results_final.csv'):
					   df.to_csv('results_final.csv', index = False, header=True, encoding = 'utf-8')
					else: # else it exists so append without writing the header
					   df.to_csv('results_final.csv', index = False, mode='a', header=True, encoding = 'utf-8')
				img_cnt+=1
				
	
