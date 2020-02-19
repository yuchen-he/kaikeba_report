#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_show(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def padding(color,kernel,flag=0):
    """
    :param color: r,g,b color
    :param kernel: kernel size
    :param flag: padding_way
    :return: new r,g,b
    """
    height,width=color.shape
    left = int((kernel - 1) / 2)
    # 左侧和上侧的填充行数
    right = kernel - 1 - left
    # 右侧和下侧的填充行数

    if flag ==0:
        left_padding = np.zeros((height, left))
        right_padding = np.zeros((height, right))
        # color = np.hstack((left_padding, color, right_padding))
        color = np.concatenate((left_padding, color, right_padding),axis=1)
        # 左右进行填充
        up_padding = np.zeros((left,width+kernel-1,))
        down_padding = np.zeros((right,width+kernel-1))
        # color = np.vstack((up_padding, color, down_padding))
        color = np.concatenate((up_padding, color, down_padding),axis=0)
        # 上下进行填充
    elif flag==1:
        left_padding = np.tile(color[:,0].reshape(height,1),left)
        right_padding = np.tile(color[:,-1].reshape(height,1),right)
        color = np.concatenate((left_padding, color, right_padding), axis=1)
        up_padding = np.tile(color[0,:],(left,1))
        down_padding = np.tile(color[-1,:],(right,1))
        color = np.concatenate((up_padding, color, down_padding), axis=0)
    else:
        pass

    return color

def medianBlur(img, kernel, padding_way='ZEROS'):
    """img & kernel is List of List; padding_way a string
    when "REPLICA" the padded pixels are same with the border pixels
    when "ZERO" the padded pixels are zeros"""
    height,width,depth=img.shape
    # shape是图片高度x宽度x通道
    rgb = cv2.split(img)
    # 分解成rgb后对每个通道进行处理
    # 如果不填充，步幅为1的情况下卷积后的图像宽为width-kernel+1，因此填充的宽度为kernel-1
    new_img = np.zeros((3,height+kernel-1,width+kernel-1))
    if padding_way == 'ZEROS':
        for i,c in enumerate(rgb):
            new_img[i] = padding(c,kernel,0)
    elif padding_way == 'REPLICA':
        for i, c in enumerate(rgb):
            new_img[i] = padding(c,kernel,1)
    else:
        pass

    blur_img = np.zeros((3,height,width))

    for t,c in enumerate(new_img):
        # 对于每个通道的高和宽
        for i in range(height):
            for j in range(width):
                blur_img[t][i][j]= np.median(c[i:i+kernel,j:j+kernel])

    blur_img=cv2.merge(blur_img).astype('uint8')
    return blur_img


if __name__ == '__main__':
    img = np.random.randint(0,10,(10,12))
    print(padding(img,7,0))
    img = cv2.imread('noisy_lenna.jpg')
    my_show(img)
    plt.show()
    newimg = medianBlur(img,7,padding_way='ZEROS')
    my_show(newimg)
    plt.show()