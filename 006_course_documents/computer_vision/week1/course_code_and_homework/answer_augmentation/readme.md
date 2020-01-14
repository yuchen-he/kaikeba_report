A random augmentation can be realized by using the augmentation module.

Ths usage is shown as below.
// a sample program main.py is included in this zip file, you can exeute it by "python main.py"

========================================
from augmentation import *

// set a margin for random.randint to generate random values for augmentation
random_margin = 30
img_orig = cv2.imread('./lenna.jpg', 1)
img_aug = data_augmentation(img_orig, random_margin)
show_img(img_aug)

========================================