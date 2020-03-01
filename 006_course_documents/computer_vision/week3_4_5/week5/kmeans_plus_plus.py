"""
1. Generate centers:
    generate first point randomly (not necessary to be a point in points)
    calculate the probabilities of all rest points of being the next points (based on distance)
    generate the next point based on a probability until K

2. Calculate distance to the K points of all the rest points
   and divide them to different clustering based on distance
3. Recalculate new K centers by calculating the mean x,y
4. Redo step 2 until all K centers do not change any more
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import random

class Kmeas_plus_plus(object):
    def __init__(self, total_data, K, max_iter):
        self.total_data = total_data
        self.K = K
        self.max_iter = max_iter

    def get_prediction(self):

        points, labels = self.gen_data()
        plt.ion()
        fig, ax = plt.subplots()

        k_centers = self.init_centers(points)
        """
        K-Means迭代的条件可以有如下几个：
            · 每个聚类内部元素不在变化，这是最理想的情况了。
            · 前后两次迭代，J的值相差小于某个阈值。
            · 迭代超过一定的次数。
        """
        predicted_labels_pre = np.zeros((self.total_data))
        for i in range(self.max_iter):
            predicted_labels = self.clustering_data(k_centers, points)
            k_centers = self.update_centers(predicted_labels, points)
            plt.scatter(points[:, 0], points[:, 1], c=predicted_labels, linewidths=0.5)
            plt.title(f"ITER: {i + 1}")
            plt.pause(0.1)
            if (predicted_labels.tolist() == predicted_labels_pre.tolist()):
                print(f"The calculation ends at {i}")
                break

        plt.ioff()
        plt.show()
        return points, predicted_labels, k_centers


    def gen_data(self):
        """
        points.shape: (1000, 2)
        labels.shape: (1000,)
        """

        points, labels = sklearn.datasets.make_blobs(n_samples=self.total_data, centers=self.K,
                                                     shuffle=True)
        return points, labels

    def init_centers(self, points):
        # initiate the first center
        k_centers = np.zeros((self.K, 2))
        first_index = np.random.randint(0, self.total_data)
        k_centers[0, ] = points[first_index, ]

        distance = np.zeros((self.total_data, 1))
        sum_dist = 0
        # select the next centers
        for i in range(1, self.K):
            # for all points, calculate the distance to the nearest center
            for j in range(self.total_data):
                min_dist, min_index = self.nearest_dist(points[j,], k_centers[0:i, ])
                distance[j] = np.power(min_dist, 2)
                sum_dist += distance[j]

            # calculate the probability of being selected
            prob = distance / sum_dist
            rand_data = random.random()
            next_index = 0
            for m, prob_m in enumerate(prob):
                if m == 0:
                    prob_m = prob_m
                else:
                    prob_m = prob_m + prob[m-1]
                # judge if rand_date in this range
                if (rand_data-prob_m) < 0:
                    next_index = m
                    break
            k_centers[i, ] = points[next_index]
        return k_centers

    def nearest_dist(self, point, exist_centers):
        min_dist = 1000000
        min_index = 0
        for i in range(exist_centers.shape[0]):
            dist = np.linalg.norm(point - exist_centers[i, ])
            if dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index

    def clustering_data(self, k_centers, points):
        """
        Return: predicted_cluster.shape(total_data,1)
        """
        predicted_labels = np.zeros(self.total_data)
        for i in range(self.total_data):
            min_dist, min_index = self.nearest_dist(points[i, ], k_centers)
            predicted_labels[i] = min_index

        return predicted_labels

    def update_centers(self, predicted_labels, points):
        predicted_cluster = []
        new_centers = np.zeros((self.K, 2))
        for i in range(self.K):
            # import pdb; pdb.set_trace()
            predicted_cluster.append(points[predicted_labels[:] == i])
            if len(predicted_cluster[i]) != 0:
                new_centers[i, ] = np.mean(predicted_cluster[i][:, 0]), np.mean(predicted_cluster[i][:, 1])
        # print(f"new_ceters={new_centers}")
        return new_centers


if __name__ == '__main__':
    total_data = 1000
    K = 4
    max_iter = 10
    kpp = Kmeas_plus_plus(total_data, K, max_iter)
    points, predicted_labels, new_centers = kpp.get_prediction()
    plt.scatter(points[:, 0], points[:, 1], c=predicted_labels, linewidths=0.5)
    plt.scatter(new_centers[:, 0], new_centers[:, 1], c='red', marker="*", linewidths=6)
    plt.show()