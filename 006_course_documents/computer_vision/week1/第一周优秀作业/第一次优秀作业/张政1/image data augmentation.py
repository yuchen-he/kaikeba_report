import cv2
import numpy as np
import matplotlib.pyplot as plt


#complete a data augmentation
#image_show(image)
def image_show(image, label):
    figure_name = ["image crop", "color shift", "gamma change", "image flip", "image similarity transform (rotation, scale, translation)", "affine transform", "perspective transform"]
    plot_title = ["image crop", "color shift", "gamma change", "image flip", "image similarity transform (rotation, scale, translation)", "affine transform", "perspective transform"]
    plt.figure(figure_name[label], figsize=(8, 8), dpi=120)
    plt.title(plot_title[label])
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


#image crop
def image_crop(image):
    print("*" * 16 + "图片切割" + "*" * 16)
    print("图片的宽为：{0[1]}，高为{0[0]}，通道数为{0[2]}".format(list(image.shape)))
    point = eval(input("请输入希望切割的新图片中左上角的第一个像素坐标,形如（x,y）："))
    width = eval(input("请输入希望显示的新图片宽度："))
    height = eval(input("请输入希望显示的宽度："))
    height_left = point[0]
    height_right = point[0] + width
    width_left = point[1]
    width_right = point[1] + height
    img_crop = image[height_left:height_right, width_left:width_right]
    image_show(img_crop, 0)
    return img_crop


#color shift
def image_color_shift(image):
    print("*" * 16 + "图片颜色转换" + "*" * 16)
    B, G, R = cv2.split(image)
    print("图片的B值为：{0[1]}，G值为{0[0]}，R值为{0[3]}".format(B, G, R))
    key = list(map(int, input("请输入需要增加（输入正值）或减少（输入负值）的B,G,R通道灰度值，并用空格分隔开：").split()))
    b_variation, g_variation , r_variation = key[0], key[1], key[2]
    if b_variation > 0:
        b_limit = 255 - b_variation
        B[B > b_limit] = 255
        B[B <= b_limit] = (B[B <= b_limit] + b_variation).astype("uint8")
    else:
        B[B > abs(b_variation)] = B[B > abs(b_variation)] + b_variation
        B[B <= abs(b_variation)] = 0
    if g_variation > 0:
        g_limit = 255 - g_variation
        G[G > g_limit] = 255
        G[G <= g_limit] = (G[G <= g_limit] + g_variation).astype("uint8")
    else:
        G[G > abs(g_variation)] = G[G > abs(g_variation)] + g_variation
        G[G <= abs(g_variation)] = 0
    if r_variation > 0:
        r_limit = 255 - r_variation
        R[R > r_limit] = 255
        R[R <= r_limit] = (R[R <= r_limit] + r_variation).astype("uint8")
    else:
        R[R > abs(r_variation)] = R[R > abs(r_variation)] + r_variation
        R[R <= abs(r_variation)] = 0
    img_color_shift = cv2.merge((B, G, R))
    image_show(img_color_shift, 1)
    return img_color_shift


#gamma change
def image_gamma_change(image, gamma=1.0):
    print("*" * 16 + "图片gamma变换" + "*" * 16)
    gamma = eval(input("请输入gamma系数值："))
    gamma_inv = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i/255.0)**gamma_inv)*255)
    table = np.array(table).astype("uint8")
    image_brighter = cv2.LUT(image, table)
    image_show(image_brighter, 2)
    return image_gamma_change


#image flip
def image_flip(image):
    print("*" * 16 + "图片翻转" + "*" * 16)
    while True:
        key = eval(input("请输入翻转模式，上下翻转请输入0，左右翻转请按1，上下左右同时翻转请按-1："))
        if key == 0:
            img_flip = cv2.flip(image, 0)  # 上下翻转
            break
        elif key == 1:
            img_flip = cv2.flip(image, 1)  # 左右翻转
            break
        elif key == -1:
            img_flip = cv2.flip(image, -1)  # 上下、左右翻转
            break
        else:
            print("未按要求赋值")
    image_show(img_flip, 3)
    return img_flip


#image similarity transform（rotation, scale, translation）
def image_similarity_transform(image):
    print("*"*16 + "图片的相似变换---(旋转，放缩，平移)" + "*"*16)
    translation = list(map(int, input("请输入图片的平移距离,请用空格将水平位移x,垂直位移y的值隔开：").split()))
    scale = eval(input("请输入图片的放缩尺度："))
    center = list(map(float, input("请输入图片的旋转中心的坐标值，请用空格将x,y的值隔开：").split()))
    rotation_angle = eval(input("请输入图片的旋转角度："))
    M = cv2.getRotationMatrix2D((center[0], center[1]), rotation_angle, scale)
    img_transform = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    M_translation = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    img_transform = cv2.warpAffine(img_transform, M_translation, (image.shape[1], image.shape[0]))
    image_show(img_transform,4)
    return img_transform


#affine transform
def image_affine_transform(image):
    print("*"*16 + "图片的仿射变换" + "*"*16)
    translation = input("请输入三对点（x,y）,并请用空格将每组（x.y）隔开：").split()
    pts1 = np.float32([list(eval(translation[0])), list(eval(translation[1],)), list(eval(translation[2]))])
    pts2 = np.float32([list(eval(translation[3])), list(eval(translation[4])), list(eval(translation[5]))])
    M = cv2.getAffineTransform(pts1, pts2)
    img_affine_transform = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    image_show(img_affine_transform, 5)
    return img_affine_transform


#perspective transform
def perspective_transform(image):
    print("*" * 16 + "图片的投射变换" + "*" * 16)
    translation = input("请输入四对点（x,y）,并请用空格将每组（x.y）隔开：").split()
    pts1 = np.float32([list(eval(translation[0])), list(eval(translation[1])), list(eval(translation[2])),
                       list(eval(translation[3]))])
    pts2 = np.float32([list(eval(translation[4])), list(eval(translation[5])), list(eval(translation[6])),
                       list(eval(translation[7]))])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective_transform = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    image_show(img_perspective_transform, 6)
    return img_perspective_transform


def augmentation(image):
    print("*" * 22 + "欢迎使用图片增广APP" + "*" * 22)
    while True:
        print("{0:*^59}".format("图片切割请按1"))
        print("{0:*^56}".format("图片颜色转换请按2"))
        print("{0:*^55}".format("图片gamma变换请按3"))
        print("{0:*^59}".format("图片翻转请按4"))
        print("{0:*^42}".format("图片相似变换（平移，旋转，放缩）请按5"))
        print("{0:*^56}".format("图片仿射变换请按6"))
        print("{0:*^56}".format("图片投射变换请按7"))
        print("{0:*^53}".format( "进行以上所有操作请按0"))
        key = list(map(int, input("请输入想要做的图片操作，并用空格将每个操作对应的数值隔开：").split()))
        for i in range(len(key)):
            if key[i] == 1:
                image_crop(image)
            elif key[i] == 2:
                image_color_shift(image)
            elif key[i] == 3:
                image_gamma_change(image)
            elif key[i] == 4:
                image_flip(image)
            elif key[i] == 5:
                image_similarity_transform(image)
            elif key[i] == 6:
                image_affine_transform(image)
            elif key[i] == 7:
                perspective_transform(image)
            elif key[i] == 0:
                image_crop(image)
                image_color_shift(image)
                image_gamma_change(image)
                image_flip(image)
                image_similarity_transform(image)
                image_affine_transform(image)
                perspective_transform(image)
                continue
            else:
                print("输入值不符合要求")
        break


if __name__ == '__main__':
    img = cv2.imread(".\\image\\lenna.jpg", 1)
    # show img
    plt.figure("image_origin", figsize=(8, 8), dpi=120)
    plt.title("image_origin")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
    #realize augmentation operation for img
    augmentation(img)