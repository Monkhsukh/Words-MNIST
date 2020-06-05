import base64
import requests
import cv2	
import numpy as np
import json
import glob, os
import ast
import pandas as pd
import unicodecsv as csv
from pathlib import Path

from lxml.html import fromstring
from itertools import cycle
import traceback

def opening(filename, kernel, savename):
	img = cv2.imread(filename,0)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	cv2.imwrite(savename,opening)

def closing(filename, kernel, savename):
	img = cv2.imread(filename,0)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite(savename,closing)

def erosion(filename, kernel, savename):
	img = cv2.imread(filename,0)
	erosion = cv2.erode(img,kernel,iterations = 1)
	cv2.imwrite(savename,erosion)

def ocr_space_request(filename, proxy_pool, proxy, api_key, engineType = 1):

    content_type = filename.split('.')[-1]
    
    img = cv2.imread(filename,0)

    encoded_string = base64.b64encode(cv2.imencode('.jpeg', img)[1]).decode()

    overlay = False
    language = 'eng'

    payload = {'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
        'scale': True,
        'OCRengine': engineType,
        'base64Image':"data:image/{};base64,{}".format(content_type,
                                                        encoded_string)
        }

    success = False

    while (not success):
        try :
            print proxy
            r = requests.post('https://api.ocr.space/parse/image',
                            data=payload, 
                            #proxies={"http": proxy, "https": proxy}, 
                            timeout= 2) 
            success = True
        except:
            proxy = next(proxy_pool)

    m = r.content.decode('utf-8')
    jsonstr = json.loads(m)
    
    try:
    	return jsonstr["ParsedResults"][0]["ParsedText"]
    except IndexError:
    	print m
        if engineType == 1 :
            ocr_space_request(filename, 2)
        return "N/A"
    except TypeError:
        print("API expired")
        return "N/A"

def writeToFile(row_list):
    if Path('result.csv').is_file():
        f = open("result.csv", "a")
        sniffer = csv.Sniffer()
        csv_dialect = sniffer.sniff(
            open("result.csv").readline())

        writer = csv.writer(f, encoding='UTF-8', quoting = csv.QUOTE_NONE, escapechar='\\', dialect=csv_dialect)
        writer.writerows(row_list)
        f.close()
    else:
        with open('result.csv','w') as file:
            writer = csv.writer(file,encoding='UTF-8', delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerows(row_list)        

def get_proxies():
    url = "https://www.sslproxies.org/"
    #url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies


proxies = get_proxies()
proxy_pool = cycle(proxies)
proxy = next(proxy_pool)
print proxies

with open('dataset/v011_labels_small.json') as json_file:
    data = json.load(json_file)

df = pd.DataFrame.from_dict(data, orient="index")

image_files = [f for f in sorted(glob.glob("dataset/dataset/v011_words_small/*.*"))]

row_list = [] #[["Path", "Actual word", "Prediction"]]

cnt = 0

print pd.read_csv('fix.csv', header=None)
mydict = pd.read_csv('fix.csv', header=None).set_index(0)[2].to_dict()

print mydict

#api_key = "7c7bc0979888957" # mr.eterna
#api_key3 = "3332508e2188957" # dexepi5603@ualmail.com
api_key1 = "fb80e2d4bb88957" # blabalbla@svpmail.com
api_key2 = "41d911225a88957" # blabalbla@itiomail.com
api_key = "3332508e2188957"

for x in image_files:
    cnt+=1
    img_name = x.split('/')[-1]
    print cnt, img_name
    if cnt > 3516 :
        #print mydict[img_name]
        #if pd.isna(mydict[img_name]):
        prediction = ocr_space_request(x, proxy_pool, proxy, api_key).rstrip('\r\n')
        if prediction == 'N/A':
            if api_key == api_key1 : 
                api_key = api_key2
            else : 
                api_key = api_key1
            prediction = ocr_space_request(x, proxy_pool, proxy, api_key).rstrip('\r\n')
        #else:
        #    prediction = mydict[img_name]
        print data[img_name], " => ", prediction
        row_list.append([img_name, data[img_name], prediction])
        
        if cnt % 100 == 0:
            print row_list
            writeToFile(row_list)
            row_list = []
            proxies = get_proxies()
            proxy_pool = cycle(proxies)