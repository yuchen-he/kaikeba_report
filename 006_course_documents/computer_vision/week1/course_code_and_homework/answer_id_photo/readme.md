This document shows how to use the id_color_change module.

■There are 6 color_change functions in this module:
  -blue2red()   : Change the blue background to red of an id photo
  -blue2white() : Change the blue background to white of an id photo
  -white2red()  : Change the white background to red of an id photo
  -white2blue() : Change the white background to blue of an id photo
  -red2white()  : Change the red background to white of an id photo
  -red2blue()   : Change the red background to blue of an id photo

■The basic usage of changing the red background to blue is shown as below:
// a sample program main.py is included in this zip file, you can exeute it by "python main.py"

##############################################
from id_color_change.py import *
import cv2

img_red  = cv2.imread('path_to_image', 1)
img_blue = red2blue(img_red)
cv2.imwrite('path_to_saved_image', img_blue)
##############################################
