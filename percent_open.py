# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import cv2

import glob, os
from pathlib import Path
import unicodecsv as csv

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
print("Setup Complete")


def writeToFile(row_list):
    file_name = 'percents.csv'
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

def compare_columns(data, col1, col2):
	comparison_column = np.where(data[col1] == data[col2], True, False)
	correct = np.count_nonzero(comparison_column) 
	total = data[col1].count()	
	#print total 
	#print correct
	#print correct / float(total)
	 
	if float(total) == 0 : 
		return 0 
	else: 
		return correct / float(total)

results = [['Otsu', 'Percent']]

#result OCR.space
# 1281 1745
# 73.4%
data = pd.read_csv("results_otsu.csv") 

kernel_types = [cv2.MORPH_RECT] #, cv2.MORPH_ELLIPSE]
#headers = ["Original","Opening", "Closing", "Erosion", "Dilation", "OC", "CO","OCO", "COC", "Sharpened"]

headers = ["Result"]
#for kernel_size in xrange(2,8):
for kernel_type in kernel_types:
	#print data[(data['Kernel size']== kernel_size) & (data['Kernel type'] == kernel_type)]
	# data_filtered = data[(data['Kernel size']== kernel_size) & (data['Kernel type'] == kernel_type)]
	for header in headers:
		results.append([header, compare_columns(data, 'Actual word', header)])

df = pd.DataFrame(data=results[1:],    # values
              columns=results[0])


# iter = df.groupby(['number_of_iteration']).mean()['Percent']
# y_pos = range(1, len(iter) +1)
# print iter


# k1 = df[df.number_of_iteration == 2].groupby(['kernel_type', 'kernel_size'], as_index=False).mean()
# print k1
# k1.kernel_type = k1.kernel_type.replace([0,1,2],['Rectangle', 'Cross', 'Ellipse'])
# plt.figure(figsize=(10, 6))
# sns.barplot(x="kernel_size", hue="kernel_type", y="Percent", data=k1)
# plt.show()

# plt.bar(y_pos, iter, align='center', alpha=0.5)
# plt.xticks(y_pos)
# plt.ylabel('%')
# plt.title('Number of iteration')

# plt.show()

# print df.groupby(['number_of_iteration','kernel_size', 'kernel_type','Morphology']).mean()['Percent']
# k1 = df[(df.number_of_iteration == 1) & ((df.kernel_size == 2) |  (df.kernel_size == 3))].groupby(['kernel_size', 'Morphology','kernel_type'], as_index=False).mean()
# print k1
# k2 = df[((df.kernel_size == 2) |  (df.kernel_size == 3))].groupby(['kernel_size', 'Morphology','kernel_type'], as_index=False).mean()
# print k2

# k2.kernel_type = k2.kernel_type.replace([0,1,2],['Rectangle', 'Cross', 'Ellipse'])
# g = sns.catplot(x="Morphology", hue="kernel_type", y="Percent", data=k2, col='kernel_size', kind='bar' )
# g.despine(left=True)
# plt.show()

print df
# print df.groupby(['Morphology', 'kernel_type']).mean()['Percent']

# print df.groupby(['Morphology']).mean()

# k1 = data[((data['Kernel size'] == 2) |  (data['Kernel size'] == 3))]
# data_null = k1[k1['Original'].isnull()]
# # print data_null

# results_perf = [["number_of_iteration", "kernel_size", "kernel_type",'Morphology', 'Percent']]
# headers = ["Opening", "Closing", "Erosion", "Dilation", "OC", "CO","OCO", "COC"]
# for number_of_iteration in xrange(1,2):
# 	for kernel_size in xrange(2,4):
# 		for kernel_type in kernel_types:
# 			data_filtered = data_null[(data_null['Kernel size']== kernel_size) & (data_null['Kernel type'] == kernel_type) & (data_null['Iterations'] == number_of_iteration)]
#  			for header in headers:
#  				results_perf.append([number_of_iteration, kernel_size, kernel_type, header, compare_columns(data_filtered, 'Actual word', header)])

# df_perf = pd.DataFrame(data=results_perf[1:],    # values
#               columns=results_perf[0])
# print df_perf
# # print df_perf.groupby(['number_of_iteration']).mean()['Percent']
# print df_perf.groupby(['kernel_size']).mean()['Percent']
# print df_perf.groupby(['kernel_type']).mean()['Percent']


# k2 = df[(df.number_of_iteration == 1) & ((df.kernel_size == 2) |  (df.kernel_size == 3))].groupby([ 'Morphology','kernel_size'], as_index=False).mean()
# print k2
# g = sns.barplot(x="Morphology", hue="kernel_size", y="Percent", data=k2)
# plt.show()


# print 'test'
# print df_perf[(df_perf.number_of_iteration == 1) & (df_perf.kernel_size <= 4)].groupby(['kernel_size', 'kernel_type', 'Morphology']).mean()['Percent']

# print df_perf.groupby(['Morphology', 'kernel_type']).mean()['Percent']

# print df_perf.groupby(['Morphology']).mean()

# data_next = pd.read_csv("results2.csv") 

# results_next = [["kernel_size",'Morphology', 'Percent']]
# headers = ["Original","Opening", "Closing", "Erosion", "Dilation", "OC", "CO","OCO", "COC"]
# for kernel_size in [1,2]:
# 	data_filtered = data_next[(data_next['Kernel size']== kernel_size)]
# 	for header in headers:
# 		results_next.append([kernel_size, header, compare_columns(data_filtered, 'Actual word', header)])

# df_next = pd.DataFrame(data=results_next[1:],    # values
#               columns=results_next[0])

# print df_next

# data_next = data_next[data_next['Original'].isnull()]

# results_next = [["kernel_size",'Morphology', 'Percent']]
# headers = ["Original","Opening", "Closing", "Erosion", "Dilation", "OC", "CO","OCO", "COC"]
# for kernel_size in [1,2]:
# 	data_filtered = data_next[(data_next['Kernel size']== kernel_size)]
# 	for header in headers:
# 		results_next.append([kernel_size, header, compare_columns(data_filtered, 'Actual word', header)])

# df_next = pd.DataFrame(data=results_next[1:],    # values
#               columns=results_next[0])

# print df_next

# sns.kdeplot(data=df['Percent'], label="Iris-setosa", shade=True)
# sns.kdeplot(data=df['Percent'], label="Iris-versicolor", shade=True)
# sns.kdeplot(data=df['Percent'], label="Iris-virginica", shade=True)

#sns.scatterplot(x=df['kernel_size'], y=df['Percent'])

#plt.show()

#writeToFile(results)


# objects = ('OCR.space', 'Tesseract', 'Opening', 'Closing', 'Erosion')
# y_pos = np.arange(len(objects))

# plt.bar(y_pos, results, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('%')
# plt.title('OCR results')

# plt.show()