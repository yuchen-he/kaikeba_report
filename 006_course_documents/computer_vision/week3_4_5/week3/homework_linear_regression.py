import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
import random
import time

# inference
def inference(w,b,x_batch):
    # Be careful about the shape when doing matrix calculation
    y_batch_pred = w * x_batch + b
    return y_batch_pred

# Calculate gradients for a batch_data
def gradient(y_batch_pred, y_batch_gt, x_batch):
    diff = y_batch_pred - y_batch_gt
    dw_list = diff * x_batch
    db_list = diff
    
    #Calculate the sum of diff_list
    dw = np.sum(dw_list)
    db = np.sum(db_list)
    return dw, db

## Use numpy to calculate gradient for a batch in one time
def cal_step_gradient(x_batch, y_batch_gt, w, b ,lr):
    dw = 0
    db = 0
    
    y_batch_pred = inference(w, b, x_batch)
    dw, db = gradient(y_batch_pred, y_batch_gt, x_batch)
    
    w -= lr*dw/len(x_batch)
    b -= lr*db/len(x_batch)
    
    return w,b

def gen_data():
    w = random.randint(0,10) + random.random()
    b = random.randint(0, 5) + random.random()
    print('w_gt:{0},b_gt:{1}'.format(w,b))
    
    x_list = np.random.randint(0, 100, 100) * np.random.rand(100)
    y_list = w * x_list + b + np.random.rand(100)*np.random.randint(-100, 100, 100)

    return x_list, y_list

def eval_loss(w,b,x_list,y_list):
    loss_list  = 0.5 * (w * x_list + b - y_list) **2
    avg_loss = np.sum(loss_list)
    avg_loss /= len(x_list)
    return avg_loss

def train_batch(x_list, y_list, batch_size, lr, num_epoch):
    w = 0
    b = 0
    
    for i in range(num_epoch):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        x_batch    = x_list[batch_idxs]
        y_batch_gt = y_list[batch_idxs]
        w, b = cal_step_gradient(x_batch, y_batch_gt, w, b ,lr)
        print('w:{0},b:{1}'.format(w,b))
        print("loss: {}".format(eval_loss(w,b,x_list,y_list)))
        
        # Show liner model at present
        plt.pause(0.1)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(np.array([0,99]), inference(w, b, np.array([0,99])), 'r-', label='Loss={}'.format(eval_loss(w,b,x_list,y_list)), lw=5)
        ax.legend()
    return w, b

x_list, y_list = gen_data()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_list, y_list)
plt.ion()

w, b = train_batch(x_list, y_list, 100, 0.001, 100)
print('w:{0},b:{1}'.format(w,b))
plt.ioff()
plt.show()
