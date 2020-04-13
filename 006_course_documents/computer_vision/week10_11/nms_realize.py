import numpy as np

def NMS(lists, thre):
    if len(lists) == 0:
        return {}
    lists = np.array(lists)
    res = []
    x1, y1, x2, y2, score = [lists[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # get sorted index in ascending order
    idxs = np.argsort(score)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        res.append(i)

        xmin = np.maximum(x1[i], x1[idxs[:last]])
        ymin = np.maximum(y1[i], y1[idxs[:last]])
        xmax = np.minimum(x2[i], x2[idxs[:last]])
        ymax = np.minimum(y2[i], y2[idxs[:last]])
        w = max(0, xmax - xmin + 1)
        h = max(0, ymax - ymin + 1)
        inner_area = w * h
        iou = inner_area / (area[i] + area[idxs[:last]] - inner_area)

        idxs = np.delete(idxs,
                         np.concatenate(([last],
                                         np.where(iou > thre)[0])))
                                         # here "where" will return us a tuple
                                         # [0] means to extract array from a tuple
