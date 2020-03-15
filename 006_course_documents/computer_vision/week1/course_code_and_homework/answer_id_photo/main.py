from id_color_change import *
import cv2

img_blue  = cv2.imread('./unnamed.jpg', 1)
img_white = blue2white(img_blue)
cv2.imwrite('./id_heyuchen.jpg', img_white)
