import numpy as np
import cv2


class Image_Stitching():
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size = 100

    # 获取单应性矩阵
    def registration(self, img1, img2):

        # 分别获取两张图的关键点kp1，kp2以及是描述子des1，des1
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # 匹配，可用BFmatcher以及FlannBasedMatcher两种方法,面对大数据集时快速最近邻搜索包（FLANN）的效果要好于 BFMatcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(des1, des2, k=2)

        good_points = []  # 用于生成单应性矩阵
        good_matches = []  # 用于画线
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        # 画线
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=0)

        # 输出图
        cv2.imwrite('matching.jpg', img3)

        # 如果确定可以拼合，获取单应性矩阵
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)

        return H

    # 定义mask矩阵，用于图像的平滑处理
    def create_mask(self, img1, img2, version):

        # 定义最终图的最大尺寸
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # 区分左右图，按照窗口尺寸进行平滑处理
        barrier = img1.shape[1] - self.smoothing_window_size
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier:width_img1] = np.tile(np.linspace(1, 0, self.smoothing_window_size).T, (height_panorama, 1))
            mask[:, :barrier] = 1
        else:
            mask[:, barrier:width_img1] = np.tile(np.linspace(0, 1, self.smoothing_window_size).T, (height_panorama, 1))
            mask[:, width_img1:] = 1

        return cv2.merge([mask, mask, mask])

    # 拼图
    def blending(self, img1, img2):

        # 获取单应性矩阵
        H = self.registration(img1, img2)

        # 定义最终尺寸
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # 对左边图像进行平滑处理
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1

        # 对右边图像进行变换，并对变换后的图像进行平滑处理
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2

        # 叠加到左边图像
        result = panorama1 + panorama2

        # 最大限度地去掉黑边
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result


def main(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final = Image_Stitching()
    final = final.blending(img1, img2)
    cv2.imwrite('panorama.jpg', final)


if __name__ == '__main__':
    try:
        main('1.jpg', '2.jpg')
    except IndexError:
        print("请输入两张图片的来源 ")
        print(
            "例如：'/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
