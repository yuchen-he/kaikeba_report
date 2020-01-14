from id_color_change import *
import cv2

img_blue  = cv2.imread('./id_photo_blue.jpg', 1)
img_red = blue2red(img_blue)
cv2.imwrite('./id_photo_blue_2_red.jpg', img_red)
