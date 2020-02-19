#Author:Wu Chao
__Author__="吴超"

#导入库
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

#导入scikit-learn鸢尾花数据集
iris = datasets.load_iris()
print(dir(iris))
print(iris.data.shape)
print(iris.feature_names)
print(iris.target.shape)
print(np.unique(iris.target))
idx= (iris.target != 2) #二分法，只考虑标签为0和1的样点
myData = iris.data[idx].astype(np.float32)
myLabel = iris.target[idx].astype(np.float32)
print(myData[:,0])#花萼长
print(myData[:,1])#花萼宽

#QC数据集
plt.scatter(myData[:,0],myData[:,1],c=myLabel,)
plt.show()

#切分数据第1列
my_Data_x1=myData[:,0]
print(my_Data_x1)
#切分数据第2列
my_Data_x2=myData[:,1]
print(my_Data_x2)

#切分数据第1列和第2列
my_Data_X1AndX2=myData[:,0:2]
print(my_Data_X1AndX2)

#检查数据与标签样点数目是否一致
print(len(myLabel))
print(len(my_Data_x1))
print(len(my_Data_x2))

'''
/*********************************************/
Logistic Regression
/*********************************************/
'''
# Step 0: 了解 sigmoid(z)函数:
'''
1. g: R→（0,1）
2. g(0)=0.5
3. g(-∞)=0
4. g(+∞)=1
5. 将函数映射到概率
'''

# Step 1: 定义 sigmoid(z)函数:
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

# Step 2: QC sigmoid(z)函数:
def plot_sigmoid():
    x = np.arange(-10, 10, .01)
    y = sigmoid(x)

    plt.plot(x, y, color='red', lw=2)
    plt.show()

plot_sigmoid() # 调用sigmoid绘图函数

# Step 3: Initialize w,b:
weights = np.zeros((1, 2)) # 平面2维图形，维度为2
bias = 0
print(weights)
print(weights)

# Step 4: y=θ0+θ1*X1+θ2*X2 → z， 实现f(x)→g(z):
def Z_Model(X):
    return sigmoid(np.dot(weights, X.T) + bias) # w为θ1，θ2，b为常数偏移项
#testZ=Z_Model(my_Data_X1AndX2)
#print(testZ)

# Step 5: 模型优化，calculate dw，db，cost:
def Z_Model_optimize(w,b,X,Y):
    # Prediction
    N=len(X)
    final_result = sigmoid(np.dot(w, X.T) + b)
    Y_T = Y.T
    # Cost
    cost = (-1 / N) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    # Gradient calculation
    dw = (1 / N) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / N) * (np.sum(final_result - Y.T))
    grads = {"dw": dw, "db": db}
    return grads, cost

# Step 6: 模型训练，Update w，b:
def logistic_Regression_train(w,b,X,Y,learningRate,maxLoopNumber):
    costs = []

    plt.ion()               # Plot Interactive Start
    fig,ax=plt.subplots()   # Plot Device

    for i in range(maxLoopNumber):
        grads, cost = Z_Model_optimize(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - (learningRate * (dw.T))
        b = b - (learningRate * db)

        if (i % 100 == 0):
            costs.append(cost)
            # print("Cost after %i iteration is %f" %(i, cost))

        if i < 500:    # QC Visualization
            plt.title('iteration:{}\n w1:{:.2f} w2:{:.2f} b:{:.2f}'.format(i,w[0][0],w[0][1],b))
            plt.xlim(4,7)
            plt.ylim(1.5,4.5)
            plt.scatter(myData[:, 0], myData[:, 1], c=myLabel, )
            my_x = np.arange(4, 7, .01)
            my_y = (-b-w[0][0]* my_x)/w[0][1]
            plt.plot(my_x, my_y, 'r-', lw=5)
            plt.text(5, 3, 'Loss=%.4f' % cost,fontdict={'size':20,'color':'red'})
            plt.pause(0.1)
            ax.cla()
            plt.show()

    # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs

# Step 7: 模型监控，QC w，b，cost:
a,b,c=logistic_Regression_train(weights,bias,my_Data_X1AndX2,myLabel,learningRate=0.01,maxLoopNumber=50000)
print(c)
print(a["w"],a["b"])