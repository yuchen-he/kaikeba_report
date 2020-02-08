import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
import random
import time
import math

def gen_data():
    # generate data (x1,y1)~(xk,yk) whose lable is 0
    # generate data (xl,yl)~(xn,yn) whose lable is 1
    # randomly put these 2 groups of data to be seperate properly
    w = random.randint(0,10) + random.random()
    b = random.randint(0, 5) + random.random()
    print('w_gt:{0},b_gt:{1}'.format(w,b))
    
    x_list = np.random.randint(0, 100, 200) * np.random.rand(200)
    
    x_list_0 = np.random.choice(x_list,100) 
    y_list_0 = w*x_list_0 + b + np.random.randint(-20, 500, 100) #* random.random()
    label_0 = np.zeros(100)
    
    x_list_1 = np.random.choice(x_list,100) 
    y_list_1 = w*x_list_1 + b - np.random.randint(-20, 500, 100) #* random.random()
    label_1 = np.ones(100)
    
    data_0 = np.vstack((x_list_0, y_list_0, label_0)).T
    data_1 = np.vstack((x_list_1, y_list_1, label_1)).T

    return data_0, data_1


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


# inference
def inference(w, point_batch_xy):
    # point_batch_xy.shape: (batch_size, 2)
    # (batch_size, 3) * (3, 1) -> (batch_size, 1)
    #import pdb; pdb.set_trace()
    ones_padding = np.ones((len(point_batch_xy), 1))
    point_batch_pad = np.hstack((ones_padding, point_batch_xy))   # (batch_size, 3)  [1, x ,y]
    
    label_batch_pred = np.dot(point_batch_pad, w)       # (batch_size, 1)
    label_batch_pred_onehot = sigmoid(label_batch_pred)
    return label_batch_pred_onehot, point_batch_pad


def eval_loss(label_batch_pred_onehot, point_batch, batch_size):
    gt = point_batch[:, -1:]
    prediction = label_batch_pred_onehot
    #import pdb; pdb.set_trace()
    loss_list  = -gt*np.log(prediction+1e-5)+(1-gt)*np.log(1-prediction+1e-5)
    avg_loss = np.sum(loss_list)
    avg_loss /= batch_size
    return avg_loss


# Calculate gradients for a batch_data
def gradient(label_batch_pred_onehot, point_batch, point_batch_pad):
    #import pdb;pdb.set_trace()
    gt = point_batch[:, -1:]
    prediction = label_batch_pred_onehot
    error = prediction - gt

    # (3, batch_size) * (batch_size, 1) -> (3, 1) for [w0,w1,w2]
    dw =  np.dot(point_batch_pad.T, error)

    return dw


## Use numpy to calculate gradient for a batch in one time
def cal_step_gradient(point_batch, w, batch_size, lr):
    
    dw = []
    #import pdb; pdb.set_trace()
    label_batch_pred_onehot, point_batch_pad = inference(w, point_batch[:, 0:2])
    loss = eval_loss(label_batch_pred_onehot, point_batch, batch_size)

    dw = gradient(label_batch_pred_onehot, point_batch, point_batch_pad)
    w -= lr*dw/batch_size
    
    return w, loss


def train_batch(data_points, batch_size, num_epoch):
    '''
    data_points.shape: (200, 3) x,y,label.  label is binary
    '''
    w = np.ones((3,1))
    lr = 0.0001
    #import pdb; pdb.set_trace()
    
    for i in range(num_epoch):
        if i < (num_epoch*1/10):
            lr = 0.001
        batch_idxs = np.random.choice(len(data_points), batch_size)
        point_batch = data_points[batch_idxs]
        w, loss = cal_step_gradient(point_batch, w, batch_size, lr)
        print("Step:{0}, loss:{1}".format(i, loss))
        
        # Show present model animation
        plt.pause(0.1)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(point_batch[:, 0], (-w[0]-w[1]*point_batch[:, 0])/w[2], 'r-', label='Step: {}'.format(i),lw=2)
        ax.legend()
    print('w_final:{0}'.format(w))
    return w


data_0, data_1 = gen_data()
data_points = np.vstack((data_0, data_1))
np.random.shuffle(data_points)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data_0.T[0], data_0.T[1], c='b')
ax.scatter(data_1.T[0], data_1.T[1], c='r')
plt.ion()

w = train_batch(data_points, 100, 100)
print("w_pred:{0}, b_pred:{1}".format(-w[1]/w[2], -w[0]/w[2]))
ax.plot(data_points[:, 0], (-w[0]-w[1]*data_points[:, 0])/w[2], 'r-', lw=2)
plt.ioff()
plt.show()
