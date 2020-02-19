import cv2
import numpy as np
img = cv2.imread('noisy_lenna.jpg', 1)

#print(r[1440,1000])
print(img.shape)
#cv2.imshow('Emily', img)
#key = cv2.waitKey()
#if key == 27:
#    cv2.destroyAllWindows()


def window_sort_for_median_in_channel(channel, n, m, window_n, window_m):
    window = []
    for i in range(0, n):
        for k in range(0, m):
            towindow = channel[window_n + i - int(n / 2), window_m + k - int(m / 2)]
            window.append(towindow)
    for i in range(0, len(window)):
        j = 1
        smallest = window[i]
        while j < len(window)-i:
            if window[i+j] < smallest:
                smallest = window[i+j]
            j += 1
        window[i] = smallest
#    print(window)
    median_window_nm = window[int(((n * m - 1) / 2))]
    return median_window_nm

#    return channel_sorted_in_window_nm


def median_blur(img, kernel, padding_way):
    b, g, r = cv2.split(img)
    height, width, channel = img.shape
    n, m = kernel.shape
    pad_r = np.zeros((height + 2 * int(n / 2), width + 2 * int(m / 2)), dtype=int)
    pad_g = np.zeros((height + 2 * int(n / 2), width + 2 * int(m / 2)), dtype=int)
    pad_b = np.zeros((height + 2 * int(n / 2), width + 2 * int(m / 2)), dtype=int)

    if padding_way == 'replica':
        for i in range(0, height + n - 1):
            for j in range(0, width + m - 1):
                if i < int(n / 2):
                    if j < int(m / 2):
                        pad_r[i][j] = r[0][0]
                        pad_g[i][j] = g[0][0]
                        pad_b[i][j] = b[0][0]
                    else:
                        if j >= width + int(m / 2):
                            pad_r[i][j] = r[0][width - 1]
                            pad_g[i][j] = g[0][width - 1]
                            pad_b[i][j] = b[0][width - 1]
                        else:
                            pad_r[i][j] = r[0][j - int(m / 2)]
                            pad_g[i][j] = g[0][j - int(m / 2)]
                            pad_b[i][j] = b[0][j - int(m / 2)]
                else:
                    if i >= height + int(n / 2):
                        if j < int(m / 2):
                            pad_r[i][j] = r[height - 1][0]
                            pad_g[i][j] = g[height - 1][0]
                            pad_b[i][j] = b[height - 1][0]
                        else:
                            if j >= width + int(m / 2):
                                pad_r[i][j] = r[height - 1][width - 1]
                                pad_g[i][j] = g[height - 1][width - 1]
                                pad_b[i][j] = b[height - 1][width - 1]
                            else:
                                pad_r[i][j] = r[height - 1][j - int(m / 2)]
                                pad_g[i][j] = g[height - 1][j - int(m / 2)]
                                pad_b[i][j] = b[height - 1][j - int(m / 2)]
                    else:
                        if j < int(m / 2):
                            pad_r[i][j] = r[i - int(n / 2)][0]
                            pad_g[i][j] = g[i - int(n / 2)][0]
                            pad_b[i][j] = b[i - int(n / 2)][0]
                        else:
                            if j >= width + int(m / 2):
                                pad_r[i][j] = r[i - int(n / 2)][width - 1]
                                pad_g[i][j] = g[i - int(n / 2)][width - 1]
                                pad_b[i][j] = b[i - int(n / 2)][width - 1]
                            else:
                                pad_r[i][j] = r[i - int(n / 2)][j - int(m / 2)]
                                pad_g[i][j] = g[i - int(n / 2)][j - int(m / 2)]
                                pad_b[i][j] = b[i - int(n / 2)][j - int(m / 2)]

    if padding_way == 'zero':
        for i in range(0, height):
            for j in range(0, width):
                pad_r[i + int(n / 2)][j + int(m / 2)] = r[i][j]
                pad_g[i + int(n / 2)][j + int(m / 2)] = g[i][j]
                pad_b[i + int(n / 2)][j + int(m / 2)] = b[i][j]
# 得到pad后的通道数据pad_r,pad_g,pad_b
    for window_n in range(0, height):
        for window_m in range(0, width):
            nm_r = window_sort_for_median_in_channel(pad_r, n, m, window_n + int(n / 2), window_m + int(m / 2))
#            print(r[window_n, window_m],nm_r)
            r[window_n, window_m] = nm_r
            nm_g = window_sort_for_median_in_channel(pad_g, n, m, window_n + int(n / 2), window_m + int(m / 2))
            g[window_n, window_m] = nm_g
            nm_b = window_sort_for_median_in_channel(pad_b, n, m, window_n + int(n / 2), window_m + int(m / 2))
            b[window_n, window_m] = nm_b
    img_blur = cv2.merge([b, g, r])
    return img_blur


kernel = np.zeros((3, 3), int)
img_blured = median_blur(img, kernel, 'replica')
cv2.imshow('Emily_Median_Blurred', img_blured)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()