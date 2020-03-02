import requests
import cv2	
import numpy as np
import json

def ocr_space_file(filename, overlay=False, api_key='7c7bc0979888957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
	       'scale': True,
	       'OCRengine': 2,
#	       'detectOrientation': True
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    m = r.content.decode()
    jsonstr = json.loads(m)
    return jsonstr["ParsedResults"][0]["ParsedText"]
#    return m


def ocr_space_url(url, overlay=False, api_key='7c7bc0979888957', language='eng'):
    """ OCR.space API request with remote file.
        Python3.5 - not tested on 2.7
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'url': url,
               'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      data=payload,
                      )
    return r.content.decode()

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


# Use examples:
#filename='Samples/1005.jpeg'
filename='Samples/100.png'

outputFolder = 'Outputs/'
erosionFilename = outputFolder + 'erosion.png'
openingFilename = outputFolder + 'opening.png'
closingFilename = outputFolder + 'closing.png'
closingAfterOpeningFilename = outputFolder + 'closingAfterOpening.png'
openingAfterClosingFilename = outputFolder + 'openingAfterClosing.png'

kernel = np.ones((3,3),np.uint8)	

erosion(filename, kernel, erosionFilename)
opening(filename, kernel, openingFilename)
closing(filename, kernel, closingFilename)
opening(closingFilename, kernel, openingAfterClosingFilename)
closing(openingFilename, kernel, closingAfterOpeningFilename)

print('Original:\n' + ocr_space_file(filename))
print('Erosion:\n' + ocr_space_file(erosionFilename))
print('Opening:\n' + ocr_space_file(openingFilename))
print('Closing:\n' + ocr_space_file(closingFilename))
print('Closing after opening:\n' + ocr_space_file(closingAfterOpeningFilename))
print('Opening after closing:\n' + ocr_space_file(openingAfterClosingFilename))

#test_url = ocr_space_url(url='https://images-na.ssl-images-amazon.com/images/I/71ovNJN1URL._SL1244_.jpg')


#print(test_url)
