import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def blue2white(img_blue):
    '''
    input : blue background id_photo in BGR color space
    output: white background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if (78<= img_hsv[x, y, 0] <=124):
                img_hsv[x, y, 1] = 0
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def blue2red(img_blue):
    '''
    input : blue background id_photo in BGR color space
    output: red background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if (78<= img_hsv[x, y, 0] <=124):
                h_lim = 180 - 70
                if img_hsv[x, y, 0]<h_lim:
                    img_hsv[x, y, 0] += 70
                else:
                    img_hsv[x, y, 0] = 180
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def red2blue(img_red):
    '''
    input : red background id_photo in BGR color space
    output: blue background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if (0<=img_hsv[x, y, 0] <=5) & (200<=img_hsv[x, y, 1] <=255):
                h_lim = 124 - 100
                if img_hsv[x, y, 0]<h_lim:
                    img_hsv[x, y, 0] += 100
                else:
                    img_hsv[x, y, 0] = 124
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def red2white(img_red):
    '''
    input : red background id_photo in BGR color space
    output: white background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if (0<=img_hsv[x, y, 0] <=5) & (200<=img_hsv[x, y, 1] <=255):
                img_hsv[x, y, 1] = 0
                img_hsv[x, y, 2] = 255
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def white2red(img_white):
    '''
    input : white background id_photo in BGR color space
    output: red background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_white, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if ((0<= img_hsv[x, y, 1] <=10) & (221<= img_hsv[x, y, 2] <=255)):
                img_hsv[x, y, 0] = 180
                img_hsv[x, y, 1] = 255
                img_hsv[x, y, 2] = 255
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def white2blue(img_white):
    '''
    input : white background id_photo in BGR color space
    output: blue background id_photo in BGR color space
    '''
    img_hsv = cv2.cvtColor(img_white, cv2.COLOR_BGR2HSV)
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if ((0<= img_hsv[x, y, 1] <=10) & (221<= img_hsv[x, y, 2] <=255)):
                img_hsv[x, y, 0] = 124
                img_hsv[x, y, 1] = 255
                img_hsv[x, y, 2] = 255
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)



