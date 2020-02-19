#!/usr/bin/env python
# coding: utf-8

# In[473]:


import numpy as np
import random
import matplotlib.pyplot as plt


# # linear regression model estimation

# In[457]:


def predict(w, b, gt_x):
    return w * gt_x + b


# # loss function define

# In[13]:


def loss_function(gt_x,gt_y,w,b):
    avg_loss = 0
    pred_y = predict(w,b,gt_x)
    diff = pred_y - gt_y
    avg_loss = np.dot(diff,diff)
    avg_loss /= 2*len(gt_y)
    return avg_loss


# # gradient calculation

# In[14]:


def cal_step_gradient(batch_gt_x,batch_gt_y,w,b,lr):
    dw = np.zeros((1,5))
    db = 0
    pred_y = predict(w,b,gt_x)
    diff = pred_y - gt_y
    dw = np.dot(diff,gt_x)
    db = diff
    
    w = w - lr * dw
    b = b - lr * db
    return w,b


# # generate data

# In[421]:


def gen_sample_data():
    w = random.randint(0,10) + random.random()
    a = random.randint(0,10)
    b = 50 * np.random.randn(1,100)+ np.random.normal(a,1,(1,100))
    
    x = np.random.uniform(0,100,size=100)
    y = w * x + b

    return x,y


# In[463]:


x,y = gen_sample_data()
plt.scatter(x,y)
plt.show()


# In[464]:


def train(x,y,batch_size,lr,max_iterations):
    w,b = 0,0
    num_sample = len(x)
    
    for i in range(max_iterations):
        batch_idxs = np.random.choice(len(x),batch_size)
        batch_x = [x[j] for j in batch_idxs]
        batch_y = [y[j] for j in batch_idxs]
        w,b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        
    return w,b


# In[474]:


w,b = train(x,y,batch_size=100,lr=0.0005,max_iterations=100)


# In[472]:


plt.title("linear regression")
plt.xlim(0,100)
plt.ylim(0,1000)
plt.plot(w,b,color='r')
plt.scatter(x,y)
plt.show

