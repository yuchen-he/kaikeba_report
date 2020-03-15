import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# 1. Read pics
# img1为query image;img2为train image
def my_read(pic1, pic2):
    img1 = cv2.imread(pic1)
    img2 = cv2.imread(pic2)
    return img1, img2

def my_show(img1, img2, size=(20, 20)):
    plt.figure(figsize=size)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

def img_show(img, size=(20, 20)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# 2. Keypoint detection
def kp_detect(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img)
    kp,des = sift.compute(img,kp)
    return kp,des

# 3. Feature match
# 利用BF、BBF、flann等（选一）算法进行特征点匹配，得到所有满足条件的特征点集good
def matching(des1,des2,error):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < error*n.distance:
            good.append(m)
    return good

def draw_match(img1,kp1,img2,kp2,good):
    draw_params = dict(matchColor = (0,255,0),singlePointColor = None,flags = 2)
    img = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img

# 4. Calculate transformation matrix
# 利用Ransac方法排除误匹配点，计算出M
def ransac(query_pts,train_pts,max_rate=0.5,iters=1000,sigma=0.5):
    best_in = 0
    M_best = np.array([])
    index_best = []
    # pt1是query点集，pt2是train点集
    for i in range(iters):
        # 1.从点集good中随机选取四个匹配点，计算M_temp
        index = random.sample(range(len(query_pts)),4)
        pst1 = np.float32([query_pts[index[0]],query_pts[index[1]],
                  query_pts[index[2]],query_pts[index[3]]])
        pst2 = np.float32([train_pts[index[0]],train_pts[index[1]],
                  train_pts[index[2]],train_pts[index[3]]])
        M_temp = cv2.getPerspectiveTransform(pst1,pst2)
        # 2.利用kp1和M_temp计算出kp2_in,与实际kp2差值<error，循环计算，得到total_in
        total_in = 0
        index_temp = []
        for j in range(len(query_pts)):
            a = np.hstack((query_pts[j],np.ones(1))).reshape(3,1)
            b = np.dot(M_temp,a)
            c = b/b[2]
            pred_t = np.delete(c,-1,axis=0).reshape(1,2)
            error = np.linalg.norm(pred_t - train_pts[j])
            if error < sigma:
                total_in += 1
                index_temp.append(j)
        # 3. 迭代得到最好的total_in，得到best_M
        if total_in > best_in:
            best_in = total_in
            M_best = M_temp
            index_best = index_temp
        # 判断模型是否已满足条件
        if best_in > max_rate*len(query_pts):
            break
        # 排除误匹配干扰
        if best_in < 7:
            best_in = index_best = 0
            M_best = np.array([])
    return M_best,best_in,index_best

# 5. Draw frame
def draw_frame(img1,img2,M):
    h,w,c = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img_final = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    img_show(img_final,size = (10,10))

# 6. Demo
query_img,train_img = my_read('pic1.jpg','pic2.jpg')
my_show(query_img,train_img)

kp_q,des_q = kp_detect(query_img)
kp_t,des_t = kp_detect(train_img)
img_q_sift = cv2.drawKeypoints(query_img,kp_q,outImage=np.array([]))
img_t_sift = cv2.drawKeypoints(train_img,kp_t,outImage=np.array([]))
my_show(img_q_sift,img_t_sift)

good = matching(des_q,des_t,0.7)
if len(good)<20:
    print('No Matching')
else:
    img_m = draw_match(query_img, kp_q, train_img, kp_t, good)
    img_show(img_m)
    query_pts = np.float32([kp_q[m.queryIdx].pt for m in good])
    train_pts = np.float32([kp_t[m.trainIdx].pt for m in good])
    M_best, best_in, index_best = ransac(query_pts, train_pts, sigma=0.5)
    if index_best == 0:
        print('No Matching')
    else:
        good_in = [good[i] for i in index_best]
        img_in = draw_match(query_img, kp_q, train_img, kp_t, good_in)
        img_show(img_in)
        draw_frame(query_img, train_img, M_best)

