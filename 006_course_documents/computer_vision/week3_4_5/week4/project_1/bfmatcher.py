# coding:utf-8

import cv2
import numpy as np

# 按照灰度图像读入两张图片
img1 = cv2.imread("./crop_img.jpg")    # train_img = label_img
img2 = cv2.imread("./50_frame.jpg")    # query_img = question_img

# 获取特征提取器对象
#orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SURF_create()

# 检测关键点和特征描述
#kp1, des1 = orb.detectAndCompute(img1, None)
#kp2, des2 = orb.detectAndCompute(img2, None)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
"""
keypoint 是关键点的列表
desc 检测到的特征的局部图的列表
"""
# 获得knn检测器
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()
#matches = bf.match(des1, des2)    # for orb
matches = bf.knnMatch(des1, des2, k=2)    # for sift/surf
#print(len(matches))
count = 0

### Export the matched points in img1 and img2
match_points_pre  = []
match_points_this = []
for m, n in matches: 
#for m in matches:   # for orb
    count += 1
    img1_idx = m.queryIdx
    img2_idx = m.trainIdx
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt
    match_points_pre.append((int(x1), int(y1)))
    match_points_this.append((int(x2), int(y2)))
    #print(count, x1,y1)
    #print(count, x2,y2)

# Find homography of perspective transform from kp1 to kp2
# S is an list of 1 or 0. 1 represents that this points in inlier
M, S = cv2.findHomography(np.array(match_points_pre), np.array(match_points_this), cv2.RANSAC, 10)


#import pdb;
#pdb.set_trace()
# recollect the matched keypoints in img2, and find the edge for generating bbox
x_min, y_min, x_max, y_max = 1000, 1000, 0, 0 
S_list = S.tolist()
match_final = []
for i in range(len(S_list)):
    if S_list[i] == [1]:
        match_final.append(match_points_this[i])
        if match_points_this[i][0] < x_min:
            x_min = match_points_this[i][0]
        if match_points_this[i][1] < y_min:
            y_min = match_points_this[i][1]
        if match_points_this[i][0] > x_max:
            x_max = match_points_this[i][0]
        if match_points_this[i][1] > y_max:
            y_max = match_points_this[i][1]

#import pdb;
#pdb.set_trace()
print('Matched points: ', len(match_final))
# Draw bbox on img2
img2_bbox = cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 255), 5)
cv2.imshow("matched", img2_bbox)


### Select good points, and get the (x,y) in the img
#good = []
#for m, n in matches:
#    if m.distance < 0.75 * n.distance:
#        good.append([m])
#good.sort(key=lambda x: x[0].distance)
#
#ret_list = []
#for match in good:
#    index = match[0].trainIdx
#    point = kp2[index].pt
#    ret_list.append((int(point[0]), int(point[1])))

"""
knn 匹配可以返回k个最佳的匹配项
bf返回所有的匹配项
"""
# 画出匹配结果
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
#cv2.imshow("matches", img3)
cv2.waitKey()
cv2.destroyAllWindows()
