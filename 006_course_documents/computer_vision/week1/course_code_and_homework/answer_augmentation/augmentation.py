import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def image_crop(img, random_margin):
    '''
    This function will do a crop for an image.
    The x1,x2,y1,y1 will be randomly selected as below:
         x1: (0, random_margin)
         x2: (x_max-random_margin, x_max)
         y1: (0, random_margin)
         y2: (y_max-random_margin, y_max)
    '''
    rows, cols, ch = img.shape
    
    x1 = random.randint(0, random_margin)
    x2 = random.randint(rows-random_margin, rows)
    y1 = random.randint(0, random_margin)
    y2 = random.randint(rows-random_margin, cols)
    img = img[x1:x2, y1:y2]
    return img


def color_shift(img, random_margin):
    '''
    This function will do a color shift in different channel.
    The shift amount of r, g, b will be randomly selected from (-random_margin, random_margin)
    '''
    B,G,R = cv2.split(img)
    r_shift = random.randint(-random_margin, random_margin)
    b_shift = random.randint(-random_margin, random_margin)
    g_shift = random.randint(-random_margin, random_margin)
    
    if r_shift > 0:
        r_lim = 255 - r_shift
        R[R>r_lim] = 255
        R[R<r_lim] = (R[R<r_lim] + r_shift).astype(img.dtype)
    else:
        r_lim = r_shift
        R[R<r_lim] = 0
        R[R>r_lim] = (R[R>r_lim] - r_shift).astype(img.dtype)

    if b_shift > 0:
        b_lim = 255 - b_shift
        B[B>b_lim] = 255
        B[B<b_lim] = (B[B<b_lim] + b_shift).astype(img.dtype)
    else:
        b_lim = b_shift
        B[B<b_lim] = 0
        B[B>b_lim] = (B[B>b_lim] - b_shift).astype(img.dtype)

    if g_shift > 0:
        g_lim = 255 - g_shift
        G[G>g_lim] = 255
        G[G<g_lim] = (G[G<g_lim] + g_shift).astype(img.dtype)
    else:
        g_lim = g_shift
        G[G<g_lim] = 0
        G[G>g_lim] = (G[G>g_lim] - g_shift).astype(img.dtype)

    return cv2.merge((B, G, R))


def rotation(img, random_margin):
    '''
    This function will do a rotation around the center of the image.
    The rotation angle will be selected from (-random_margin, random_margin)
    '''
    size = (img.shape[1], img.shape[0])
    
    center = (img.shape[1]/2, img.shape[0]/2)
    angle  = random.randint(-random_margin, random_margin)
    scale  = random.uniform(0,5)
    M_rt = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotate = cv2.warpAffine(img, M_rt, size)
    return img_rotate


def perspective_transform(img, random_margin):
    '''
    This function will do a perspective_transform for an image randomly.
    '''
    rows, cols, ch = img.shape
    
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(cols - random_margin - 1, cols - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(cols - random_margin - 1, cols - 1)
    y3 = random.randint(rows - random_margin - 1, rows - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(rows - random_margin - 1, rows - 1)
    
    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(cols - random_margin - 1, cols - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(cols - random_margin - 1, cols - 1)
    dy3 = random.randint(rows - random_margin - 1, rows - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(rows - random_margin - 1, rows - 1)
    
    pts_src   = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts_target = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_pt = cv2.getPerspectiveTransform(pts_src, pts_target)
    img_warp = cv2.warpPerspective(img, M_pt, (cols, rows))
    return img_warp


def data_augmentation(img, random_margin):
    '''
    Input: 
        1. image in BGR color space
        2. specified random_margin
    '''
    img = image_crop(img, random_margin)
    img = color_shift(img, random_margin)
    img = rotation(img, random_margin)
    img = perspective_transform(img, random_margin)
    return img

def show_img(img):
    '''
    Input: image in BGR color space
    '''
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# img_orig = cv2.imread('./lenna.jpg', 1)
# img_aug = data_augmentation(img_orig)
# show_img(img_aug)
