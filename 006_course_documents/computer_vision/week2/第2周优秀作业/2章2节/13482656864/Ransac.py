#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

# 本代码由网友提供于CSDN，此处略作修改，代码注释由本人添加
# https://blog.csdn.net/vict_wang/article/details/81027730

import numpy as np
import scipy as sp
import scipy.linalg as sl
import matplotlib.pyplot as plt

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回None,如果未找到）

    iterations = 0
    bestfit = None #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k：
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)：
            if 满足maybemodel即error < t
                将点加入alsoinliers

        if (alsoinliers样本点数目 > d)：
            # 有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr：
                bestfit = bettermodel
                besterr = thiserr

        iterations +=1
    return bestfit
    """

    iterations = 0
    bestfit = None
    besterr = np.inf
    # 设置besterr默认值为无穷大
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        # 获取随机的n个下标位置
        maybe_inliers = data[maybe_idxs, :]
        # 获取maybe_idxs中所在行的数据(Xi,Yi)，即n行的随机数据
        test_points = data[test_idxs, :]
        # 获取test_idxs中所在行的数据(Xi,Yi)，即n行的随机数据
        maybemodel = model.fit(maybe_inliers)
        # 拟合模型
        test_err = model.get_error(test_points, maybemodel)
        # 计算误差:平方和最小
        also_idxs = test_idxs[test_err < t]
        # test<t返回布尔值数组，将test_idxs中满足test<t的位置取出来
        also_inliers = data[also_idxs, :]
        # 将data中的也是局内点取出来
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if len(also_inliers > d):
            # 如果局内点的数量大于阈值
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            # 将可能局内点和也是局内点连接
            bettermodel = model.fit(betterdata)
            # 再一次用模型去计算结果
            better_errs = model.get_error(betterdata, bettermodel)
            # 计算误差
            thiserr = np.mean(better_errs)
            # 将平均误差作为新的误差
            if thiserr < besterr:
                # 如果本次误差小于最佳误差
                bestfit = bettermodel
                # 最佳模型就是本次模型
                besterr = thiserr
                # 最好的误差就是本次误差
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
                # 最终的局内点在数据中心的位置
        iterations += 1
        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit


def random_partition(n, n_data):
    """return n random rows of data
    n_data is len(data)"""
    all_idxs = np.arange(n_data)
    # 获取n_data下标索引
    np.random.shuffle(all_idxs)
    # 打乱下标索引
    idxs1 = all_idxs[:n]
    # 获取打乱下标的前n个位置处的值
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型，也可以直接使用Sklearn中的线性回归模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)
        # residues:残差和
        return x
        # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)
        # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        # sum squared error per row
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500
    # 样本个数
    n_inputs = 1
    # 输入变量个数，可调
    n_outputs = 1
    # 输出变量个数，可调
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    # 随机线性度即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)
    # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)
    # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        # A_noisy.shape[0]=500，获取索引0-499
        np.random.shuffle(all_idxs)
        # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]
        # 取100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))
        # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))
    # 沿着axis=0堆叠，shape=(500,2)
    input_columns = range(n_inputs)
    # 数组的第一列x:0,rang(0,1)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    # 数组最后一列y:1,[1]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # 最小二乘法进行计算

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    # RANSAC算法进行计算
    if 1:
        sort_idxs = np.argsort(A_exact[:, 0])
        # 返回A_exact[:, 0]排序后每个元素所在的位置
        A_col0_sorted = A_exact[sort_idxs]
        # 将A_exact进行排序
        # 以上两行代码相当于sorted(A_exact[:, 0])

        # 以下进行可视化
        if 1:
            # 改成0即打印else的内容
            plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            # 将A_noisy，B_noisy中的第一列打印成散点图
            plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
            # 将RANSCAC算法的局内点打印出来
        else:
            # 以下打印局外点
            plt.plot(A_noisy[outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            plt.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        plt.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        plt.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        plt.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test()