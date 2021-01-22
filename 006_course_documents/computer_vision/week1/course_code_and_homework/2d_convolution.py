import cv2
import numpy as np

img = cv2.imread('./lenna.jpg', 0)
kernal = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

def convolution(k, data):
    height, width = data.shape
    img_new = []

    for i in range(width-3):
        line = []
        for j in range(height-3):
            tgt = data[i:i+3, j:j+3]
            line.append(np.sum(np.multiply(k, tgt)))
        img_new.append(line)
    return np.array(img_new)

img_new = convolution(kernal, img)
cv2.imwrite('./lenna_conv2d.jpg', img_new)
#cv2.namedWindow('lenna_conv2d')
#cv2.imshow('lenna_conv2d', convolution(kernal, img))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
