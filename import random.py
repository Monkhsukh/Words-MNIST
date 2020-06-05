import random
import copy
import hashlib
import io
import itertools
import os
import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pause
import pytesseract
import requests
from PIL import Image

import invoiceparser.tesseract_parser as tp
from business_objects import OCRMethod
from invoiceparser.image_util import PIL_img_to_bytes, scale_image_to_filesize
from logger_setup import getLogger
from business_objects import Fragment, InvoiceDocument, YDirection

from tempfile import TemporaryFile
from utility import RenamingUnpickler

logger = getLogger(__name__)



"""
====================================================================
Base OCR 
====================================================================
"""

CACHE_FILE_NAME = os.path.join(os.path.dirname(__file__), 'ocr_output_cache.pickle')


def _init_cache() -> Dict[Tuple[OCRMethod, str], List[Fragment]]:
    global cache

    if not os.path.isfile(CACHE_FILE_NAME):
        return {}

    with io.open(CACHE_FILE_NAME, 'rb') as f:
        return RenamingUnpickler(f).load()

def _save_cache():
    global cache

    with io.open(CACHE_FILE_NAME, 'wb') as f:
        pickle.dump(cache, f)
    
cache = _init_cache()

class OCRBase(ABC):

    def __init__(self):
        pass

    def get_fragment_list(self, image_name: str, image_bytes: bytes, use_cache: bool = False) -> List[Fragment]:
        global cache
        
        md5hash = hashlib.md5(image_bytes).hexdigest()
        key = (self._get_method(), image_name + str(md5hash))
        
        if use_cache and key in cache:
            logger.debug('Found the image in the cache by key |%s|!', key)
            return copy.deepcopy(cache[key])

        fragments = self._get_fragment_list(image_name, image_bytes)

        if use_cache and len(fragments) > 0:
            logger.debug('Adding %s to cache', '\n\t'.join(str(fragment) for fragment in fragments))
            cache[key] = fragments
            _save_cache()

        return fragments

    @abstractmethod
    def _get_fragment_list(self, image_name: str, image_bytes: bytes) -> List[Fragment]:
        pass


    @abstractmethod
    def _get_method(self) -> OCRMethod:
        pass

"""
====================================================================
Azure OCR 
====================================================================
Pricing: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/
API: https://westus.dev.cognitive.microsoft.com/docs/services/5adf991815e1060e6355ad44/operations/56f91f2e778daf14a499e1fc
"""


class AzureOCR(OCRBase):


    AZURE_PAUSE_TIME = timedelta(seconds=3)
    AZURE_OCR_URL = "https://westeurope.api.cognitive.microsoft.com/vision/v2.0/ocr"
    AZURE_SUBSCRIPTION_KEY = "44f8dd20ad21451ead1d09c87cf36e9d"
    MAX_IMAGE_SIZE = 4 * 1024 * 1024
    MAX_DIMENSION = 4000


    def __init__(self):
        self.next_azure_request_time = datetime.now()


    def _get_fragment_list(self, image_name: str, image_bytes: bytes) -> List[Fragment]:
        image_bytes = AzureOCR.__normalize_image(image_bytes)
        logger.debug(len(image_bytes))

        logger.debug('Pausing until |%s|', self.next_azure_request_time)
        pause.until(self.next_azure_request_time)# Need to throttle down as per API contract

        headers    = {'Ocp-Apim-Subscription-Key': AzureOCR.AZURE_SUBSCRIPTION_KEY,
                        'Content-Type': 'application/octet-stream'}
        params     = {'language': 'unk', 'detectOrientation': 'true'}
        response = requests.post(AzureOCR.AZURE_OCR_URL, headers=headers, params=params, data=image_bytes)
        if response.status_code != 200:
            logger.warn('Bad response |%d|! Details:\n %s', response.status_code, response.json())
        response.raise_for_status()

        self.next_azure_request_time = datetime.now() + AzureOCR.AZURE_PAUSE_TIME

        return AzureOCR.__get_fragments_from_json(response.json())

    def _get_method(self) -> OCRMethod:
        return OCRMethod.AZURE

    @staticmethod
    def __wordbox_to_fragment(wordbox) -> Fragment:
        x = int(wordbox['boundingBox'].split(',')[0])
        y = int(wordbox['boundingBox'].split(',')[1])
        width = int(wordbox['boundingBox'].split(',')[2])
        height = int(wordbox['boundingBox'].split(',')[3])
        text = ' '
        for word in [word['text'] for word in wordbox['words']]:
            # handling the case when a number's digits got separated
            if text[-1].isdigit() and word[0].isdigit():
                text = text + word
            else:
                text = text + ' ' + word
        text = text.strip()

        return Fragment(text, x, y, width, height)


    @staticmethod
    def __get_fragments_from_json(json_object) -> List[Fragment]:
        logger.debug(json_object)
        wordboxes = [region['lines'] for region in json_object['regions']]
        wordboxes = list(itertools.chain.from_iterable(wordboxes))
        fragments = map(lambda wordbox: AzureOCR.__wordbox_to_fragment(wordbox), wordboxes)
        return list(fragments)

    

    @staticmethod
    def __normalize_image(image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))

        sizefactor = float(AzureOCR.MAX_IMAGE_SIZE) / len(image_bytes)

        if img.height > img.width:
            pixelfactor = AzureOCR.MAX_DIMENSION / img.height
        else:
            pixelfactor = AzureOCR.MAX_DIMENSION / img.width

        if pixelfactor < 1.0 or sizefactor < 1.0:
            logger.info('Image is too large! Size: |%s, %d bytes| Dimensions: (%d, %d)', img.size, len(image_bytes), img.width, img.height)

            factor = min(sizefactor, pixelfactor)
            factor = factor * 0.9
            # This is hacky but the size of the output image's size from PIL is not determenistic. 
            # So if we need to resize let's be aggressive about it.

            logger.info('Reducing image to |%f| of the original size!', factor)
            img = img.resize((int(img.width * factor), int(img.height * factor)))
            logger.info('New size of the image is |%s|', img.size)

            return PIL_img_to_bytes(img)
        else:
            return image_bytes

"""
====================================================================
Tesseract OCR 
====================================================================
"""

class TesseractOCR(OCRBase):


    def _get_fragment_list(self, image_name: str, image_bytes: bytes) -> List[Fragment]:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        document = None
        with tempfile.TemporaryDirectory() as tmpdir:
            imgfname = os.path.join(tmpdir,  "{}.png".format(os.getpid()))
            cv2.imwrite(imgfname, gray)
            hocr_str = pytesseract.pytesseract.run_and_get_output(imgfname, lang=None, config="hocr", extension='hocr')
            document = tp.HOCRDocument(hocr_str)

        page=document.pages[0]
        
        l = []
        for area in page.areas:
            for paragraph in area.paragraphs:
                for line in paragraph.lines:
                    for word in line.words:
                        l.append(
                                Fragment(
                                        word.ocr_text,
                                        word.coordinates[0],
                                        4000 - word.coordinates[1],
                                        word.coordinates[2] - word.coordinates[0],
                                        word.coordinates[3] - word.coordinates[1]
                                        )
                                )
        return l


    def _get_method(self) -> OCRMethod:
        return OCRMethod.TESSERACT

"""
====================================================================
OCR.SPACE OCR 
====================================================================
"""

class OCRSpaceOCR(OCRBase):
    # wayasam@gmail.com (paid), wayasam@gmail.com (free), vszm5@hotmail.com, vszm@inf.elte.hu Get more keys if needed
    API_KEYS = ['PKMXB9465888A', 'f0a34a178b88957', '62f395656388957', '96b2d9106788957']
    MAX_SIZE_BYTES = 1024 * 1024 * 1

    def _get_fragment_list(self, image_name: str, image_bytes: bytes) -> List[Fragment]:
        image_bytes = scale_image_to_filesize(image_bytes, OCRSpaceOCR.MAX_SIZE_BYTES)        

        with TemporaryFile(suffix='.' + image_name.split('.')[-1]) as fp:
            fp.write(image_bytes)
            fp.flush()
                
            for api_key in random.sample(OCRSpaceOCR.API_KEYS, len(OCRSpaceOCR.API_KEYS)):
                fp.seek(0)
                payload = { 'isOverlayRequired': True,
                            'apikey': api_key,
                            'detectOrientation': True,
                            'language': 'hun'
                        }

                with requests.Session() as session:
                    response = session.post('https://api.ocr.space/parse/image',
                                    files={'filename': fp},
                                    data=payload,
                                    headers={'Connection':'close'})
                    
                json = response.json()

                if response.status_code == 200 and json["OCRExitCode"] < 3:
                    logger.info('Returning stuff: %s', json)
                    return self.__get_fragments_from_json(json)
                else:
                    logger.warn('Error occured: |%s|', response.text)

                logger.debug('Failed to get correct response with key |%s|, Code: |%d|, Response: |%s|',\
                        api_key, response.status_code, response.json())
                #logger.debug('Sleeping for 35 seconds for API throttling')
                #time.sleep(35)
                
                
        logger.error('Could not get response with any of the api keys. Image Name: |%s|, bytes length: |%d|',\
                        image_name, len(image_bytes))
        return []
    
    def _get_method(self) -> OCRMethod:
        return OCRMethod.OCRSPACE


    
    @staticmethod
    def __line_to_fragment(line) -> Fragment:
        x = int(line['Words'][0]['Left'])
        y = int(line['Words'][0]['Top'])
        width = int(line['Words'][-1]['Left'] + line['Words'][-1]['Width'] - x) 
        height = int(line['Words'][-1]['Top'] + line['Words'][-1]['Height'] - y)
        text = line['LineText'].strip()

        return Fragment(text, x, y, width, height)


    @staticmethod
    def __get_fragments_from_json(json_object) -> List[Fragment]:
        try:
            lines = [line for line in json_object['ParsedResults'][0]['TextOverlay']['Lines']]
            fragments = map(lambda line: OCRSpaceOCR.__line_to_fragment(line), lines)
            return list(fragments)
        except:
            logger.error('Could not parse json! |%s|', json_object)
            return []

def get_ocr_engine(method: OCRMethod) -> OCRBase:
    if method == OCRMethod.OCRSPACE:
        return OCRSpaceOCR()
    elif method == OCRMethod.AZURE:
        return AzureOCR()
    else:
        return TesseractOCR()