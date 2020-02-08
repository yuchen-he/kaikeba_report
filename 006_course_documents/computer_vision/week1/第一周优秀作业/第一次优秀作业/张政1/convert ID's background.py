import numpy as np
import cv2
import matplotlib.pyplot as plt


#身份证背面图片换底
def ID_back_cvtBackground():
    """功能：实现身份证背面更换背景色"""
    img = cv2.imread(".\\image\\ID_img.jpg")
    while True:
        try:
            color = list(map(int,  input("请输入新背景颜色的BGR值,每个数值间请用空格分开：").split()))
            break
        except ValueError:
            print("输入颜色的BGR值格式有误，请按此格式输入：B G R")
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #选择原图的背景像素作为mask中的目标像素
    aim = np.uint8([[img[0, 0, :]]])
    hsv_aim = cv2.cvtColor(aim, cv2.COLOR_BGR2HSV)
    #生成mask函数
    mask = cv2.inRange(im_hsv, np.array([hsv_aim[0, 0, 0]-60, hsv_aim[0, 0, 1]-30, hsv_aim[0, 0, 2]-50]),\
    np.array([hsv_aim[0, 0, 0]+125, hsv_aim[0, 0, 1]+35, 140]))
    img_median = cv2.medianBlur(mask, 5)     # 中值滤波，去除一些边缘噪点
    mask = img_median
    mask = cv2.erode(mask, None, iterations=2)   #腐蚀
    mask = cv2.dilate(mask, None, iterations=1)   #膨胀
    mask_inv = cv2.bitwise_not(mask)
    #保留原图除背景外的部分
    img1 = cv2.bitwise_and(img, img, mask=mask_inv)
    # 更换原图背景
    bg = img.copy()
    rows, cols, channels = img.shape
    bg[:rows, :cols, :] = color
    img2 = cv2.bitwise_and(bg, bg, mask=mask)
    #叠加img1和img2
    img_output = cv2.add(img1, img2)
    #输出图片
    image = [img,  img_output]
    title = ["img_original", "img_convert_background"]
    plt.figure("convert ID's background", figsize=(15, 5))
    for key in zip(range(2), title):
        plt.subplot(1, 3, key[0]+1)
        plt.title(key[1])
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image[key[0]], cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("mask")
    plt.axis("off")
    plt.imshow(mask, cmap="gray")
    plt.tight_layout()
    plt.show()


# 身份证正面图片换底
def ID_front_cvtBackground():
    """功能：实现身份证前面更换背景色"""
    img = cv2.imread(".\\image\\IDcard.jpg")
    while True:
        try:
            color = list(map(int,  input("请输入新背景颜色的BGR值,每个数值间请用空格分开：").split()))
            break
        except ValueError:
            print("输入颜色的BGR值格式有误，请按此格式输入：B G R")
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #选择原图的背景像素作为mask中的目标像素
    aim = np.uint8([[img[0, 0, :]]])
    hsv_aim = cv2.cvtColor(aim, cv2.COLOR_BGR2HSV)
    #生成mask函数
    mask = cv2.inRange(im_hsv, np.array([0, 0, hsv_aim[0, 0, 2]+56]),\
    np.array([hsv_aim[0, 0, 0], 48, 255]))
    img_median = cv2.medianBlur(mask, 5)     # 中值滤波，去除一些边缘噪点
    mask = img_median
    mask = cv2.erode(mask, None, iterations=2)   #腐蚀
    mask = cv2.dilate(mask, None, iterations=1)   #膨胀
    mask_inv = cv2.bitwise_not(mask)
    #保留原图除背景外的部分
    img1 = cv2.bitwise_and(img, img, mask=mask_inv)
    # 更换原图背景
    bg = img.copy()
    rows, cols, channels = img.shape
    bg[:rows, :cols, :] = color
    img2 = cv2.bitwise_and(bg, bg, mask=mask)
    #叠加img1和img2
    img_output = cv2.add(img1, img2)
    #输出图片
    image = [img,  img_output]
    title = ["img_original", "img_convert_background"]
    plt.figure("convert ID's background", figsize=(15, 5))
    for key in zip(range(2), title):
        plt.subplot(1, 3, key[0]+1)
        plt.title(key[1])
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image[key[0]], cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("mask")
    plt.axis("off")
    plt.imshow(mask, cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ID_back_cvtBackground()
    ID_front_cvtBackground()
