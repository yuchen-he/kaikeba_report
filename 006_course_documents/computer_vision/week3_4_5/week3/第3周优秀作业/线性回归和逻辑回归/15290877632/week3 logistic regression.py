#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import random
import matplotlib.pyplot as plt


# # loss function and sigmoid function define

# In[21]:


def loss_function(gt_x,gt_y,w,b):
    avg_loss = 0
    pred_y = predict(w,b,gt_x)
    diff = pred_y - gt_y
    avg_loss = np.dot(diff,diff)
    avg_loss /= 2*len(gt_y)
    return avg_loss


# In[22]:


def sigmoid(z):
    new_X = 1 / (1 + np.exp(-z))
    return new_X


# # gradient calculation

# In[23]:


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

# In[24]:


def gen_sample_data():
    w = random.randint(0,10) + random.random()
    a = random.randint(0,10)
    b = 50 * np.random.randn(1,100)+ np.random.normal(a,1,(1,100))
    
    x = sigmoid(a)
    y = w * x + b

    return x,y


# In[25]:


x,y = gen_sample_data()
plt.scatter(x,y)
plt.show()


# In[26]:


def train(x,y,batch_size,lr,max_iterations):
    w,b = 0,0
    num_sample = len(x)
    
    for i in range(max_iterations):
        batch_idxs = np.random.choice(len(x),batch_size)
        batch_x = [x[j] for j in batch_idxs]
        batch_y = [y[j] for j in batch_idxs]
        w,b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        
    return w,b


# In[27]:


w,b = train(x,y,batch_size=100,lr=0.0005,max_iterations=100)
plt.title("linear regression")
plt.xlim(0,100)
plt.ylim(0,1000)
plt.plot(w,b,color='r')
plt.scatter(x,y)
plt.show

