#导入库
import matplotlib.pyplot as plt
import numpy as np

#定义线性回归训练函数
def linear_Regression_train(data_x,data_y,learningRate,maxLoopNumber):
    N=len(data_x)
    print("sample num=",N)  # Get Sample Number
    weight = 0.0  #
    print("weight=",weight) # Initialize weight
    bias = 0.0
    print("bias=",bias)     # Initialize bias
    matrixOne=np.array(np.ones(shape=(1, N),dtype=np.float))
    print("matrixOne=", matrixOne)

    plt.ion()               # Plot Interactive Start
    fig,ax=plt.subplots()   # Plot Device

    for num in range(maxLoopNumber):# iteration for most optimized “Loss”， update “w” and “b” Parameter
        WXPlusB=data_x*weight+bias #        预测Y值 →→     （AXi+b）
        delta=WXPlusB*(-1)+data_y  # 实际Y值-预测Y值 →→ Yi-（AXi+b）
        loss = (1.0/ N) * np.dot(delta, delta.T)  # loss=∑(Ei*Ei)/N==(∑((Yi-AXi-B)*(Yi-AXi-B)))/N
        w_gradient = -(2.0/N) * np.dot(delta, data_x.T)    # dw=-2/N*∑((Yi-AXi-B)*Xi)
        b_gradient = -(2.0/N) * np.dot(delta, matrixOne.T) # db=-2/N*∑((Yi-AXi-B))
        weight = weight - w_gradient*learningRate   # y=y-△y →→ y=y-k△x →→ y=y-（dy/dx）*lr
        bias =   bias   - b_gradient*learningRate   # y=y-△y →→ y=y-k△x →→ y=y-（dy/dx）*lr

        if num % 100==0:
            print(loss) # QC Loss

        if num < 100:    # QC Visualization
            plt.title("iteration:{}\n w:{} b:{}".format(num,weight,bias))
            plt.xlim(0,5)
            plt.ylim(0,15)
            plt.plot(data_x,WXPlusB,'r-',lw=5)
            plt.scatter(data_x,data_y)
            plt.text(3, 3, 'Loss=%.4f' % loss,fontdict={'size':20,'color':'red'})
            plt.pause(0.1)
            ax.cla()
            plt.show()

    return (weight, bias)

#初始化输入数据
my_data_x=[0.5,0.6,0.8,1.1,1.4] # 测试数据1
my_data_y=[5,5.5,6,6.8,7] # Y=2.219X+4.1073
#my_data_x=[1,2,3,4,5] # 测试数据2
#my_data_y=[3,5,7,9,11] # y=2x+1
plt.scatter(my_data_x, my_data_y)
plt.show() # QC Scatter Sample
# 注释：也可以随机生成1000份数据，从随机提取一个batch作为训练，本次作业中省略

#将输入数据转为矩阵
arrayX=np.array(my_data_x)
arrayY=np.array(my_data_y)

#机器学习训练
a,b=linear_Regression_train(arrayX,arrayY,learningRate=0.01,maxLoopNumber=1000)
print("Final weight=",a)
print("Final bias=",b)