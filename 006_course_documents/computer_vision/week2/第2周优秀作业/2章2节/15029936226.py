'''
author      :   Lucas Liu
date        :   2020/1/23
description :   week 2 homework, mid level image process, including median blur method and
                related testcase
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime

'''
description:
    show the picture of the image
input:
    img             image in cv2.imread format(BGR)
output :
    None  
'''
def image_show(img, name):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(name)
    plt.show()


'''
description:
    do the midian blur operation for the given image
input:
    img             image in 1 channel
    kernel          kernel size, h * w
    padding_way     padding way, REPLICA or ZERO
    method          kernel update method, NORMAL or FAST
output :
    img_medianBlur  medianblured image
'''
def medianBlur(img, kernel, padding_way, method):
    # assert exception
    if len(img) == 0:
        print('input img is empty! return origin image!')
        return img
    if len(kernel) <= 1:
        print('kernel should be 2 demension, return origin image!')
        return img
    if (kernel[0] % 2 == 0) or (kernel[1] % 2 == 0):
        print('kernel size should be odd, return origin image!')
        return img
    
    # initialize
    h_kernel, w_kernel = kernel[:2]
    h_img = len(img[0]) # kernel height
    w_img = len(img[1]) # kernel width
    img_medianBlur = np.zeros([h_img, w_img])
    img_medianBlur = img_medianBlur.astype(int)
    r_h = h_kernel//2 # radius of kernel height
    r_w = w_kernel//2 # radius of kernel width
    img_withpadding = np.zeros([img.shape[0] + r_h * 2, img.shape[1] + r_w * 2]) # initialize image with padding
    img_withpadding = img_withpadding.astype(int)
    h_padding = img_withpadding.shape[0] # image with padding size height
    w_padding = img_withpadding.shape[1] # image with padding size width

    if (padding_way != 'REPLICA') and (padding_way != 'ZERO'):
        print('border types not supported, only suppport REPLICA or ZERO right now, return origin image!')
        return img

    # copy original image to the center of the dest matrix
    for i in range(h_img):
        for j in range(w_img):
            img_withpadding[i + r_h, j + r_w] = img[i, j]

    # padding border of the dest matrix
    for i in range(r_h):
        for j in range(w_padding):
            if padding_way == 'REPLICA':
                img_withpadding[i, j] = img_withpadding[r_h, j]
                img_withpadding[i + h_img + r_h, j] = img_withpadding[h_img, j]
            else:
                img_withpadding[i, j] = 0
                img_withpadding[i + h_img + r_h, j] = 0
    for i in range(h_padding):
        for j in range(r_w):
            if padding_way == 'REPLICA':    
                img_withpadding[i, j] = img_withpadding[i, r_w]
                img_withpadding[i, j + w_img + r_w] = img_withpadding[i, w_img]
            else:
                img_withpadding[i, j] = 0
                img_withpadding[i, j + w_img + r_w] = 0

    # kernel calculation

    # normal method, with O(h_img * w_img * h_kernel * w_kernel)
    if method == 'NORMAL':
        hist_kernel = np.zeros(h_kernel * w_kernel)
        for i in range(h_img):
            for j in range(w_img):
                for y in range(h_kernel):
                    for x in range(w_kernel):
                        hist_kernel[x + y * w_kernel] = img_withpadding[i + y, j + x]
                kernel_order = hist_kernel[:] # copy kernel to a new list
                kernel_order = kernel_order.astype(int) # change data type to int
                kernel_order.sort() # sort the list
                img_medianBlur[i, j] = kernel_order[h_kernel * w_kernel //2] # fetch the mid value to the dest pixel

    # advanced method, with O(h_img * w_img * h)
    if method == 'FAST':
        for i in range(h_img):
            l_kernel = []
            for j in range(w_img):
                
                # complete update kernel in the start of each line
                if j == 0:
                    for x in range(w_kernel):
                        for y in range(h_kernel):
                            l_kernel.append(img_withpadding[i + y, j + x])
                
                # onle need to update 1 col of the kernel, delete first col, append new col to the last col, thus we use a queue
                else:
                    for n in range(h_kernel):
                        l_kernel.remove(l_kernel[n])
                        l_kernel.append(img_withpadding[i + n, j + w_kernel - 1])

                kernel_order = l_kernel[:] # copy kernel queue to a new list
                kernel_order.sort() # sort the list
                img_medianBlur[i, j] = kernel_order[h_kernel * w_kernel //2] # fetch the mid value to the dest pixel

    return img_medianBlur

# test
img_ori = cv2.imread('./lenna.jpg', 0)
image_show(img_ori, 'img_ori')
kernel = [3, 5] # h,w

oldtime = datetime.datetime.now()
img_rep_normal = medianBlur(img_ori, kernel, 'REPLICA', 'NORMAL')
newtime = datetime.datetime.now()
print('consume for REPLICA, NORMAL:', newtime - oldtime)
image_show(img_rep_normal, 'img_rep_normal')

oldtime = datetime.datetime.now()
img_zero_normal = medianBlur(img_ori, kernel, 'ZERO', 'NORMAL')
newtime = datetime.datetime.now()
print('consume for ZERO, NORMAL:', newtime - oldtime)
image_show(img_zero_normal, 'img_zero_normal')

oldtime = datetime.datetime.now()
img_rep_fast = medianBlur(img_ori, kernel, 'REPLICA', 'FAST')
newtime = datetime.datetime.now()
print('consume for REPLICA, FAST:', newtime - oldtime)
image_show(img_rep_fast, 'img_rep_fast')

oldtime = datetime.datetime.now()
img_rep_fast = medianBlur(img_ori, kernel, 'ZERO', 'FAST')
newtime = datetime.datetime.now()
print('consume for ZERO, FAST:', newtime - oldtime)
image_show(img_rep_fast, 'img_rep_fast')
