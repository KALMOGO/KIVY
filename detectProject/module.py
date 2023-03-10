import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import time ,sys
import urllib.request
import urllib
import cv2
import numpy as np
import time
import sys
import os
import pickle

def img_improver(img=None):
    """
        impove the quality of an image through applying 
        equilisation and bilateral filter
    Args:
        img (Image): image that we want to improve the quality

    Returns:
        Image: image traited
    """
    assert img is not None, "Your image is not good !!!"
    
    img = cv2.equalizeHist(img)
    img = cv2.bilateralFilter (img,1,75, 100, borderType=cv2.BORDER_CONSTANT)
    
    return img


def resize_image(img=None, percent=0):
    """
        resize an imge to (scale) percent of his size

    Args:
        img (Image, optional): image to be resize. Defaults to None.

    Returns:
        Image: image redimensionnee 
    """
    
    assert img is not None, "ERROR: image invalid !!!"
    assert percent>0, "Error: the percent of reduction should be between 1 and 100 !!!" 
    
    scale_percent = percent # pourcentage de reduction de l'image : percent%

    #calculate the percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    return cv2.resize(img, dsize)
