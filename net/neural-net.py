import matplotlib.pyplot as plt
import numpy as np
#准备数据集
# train_x=np.array([[1]])
# train_y=np.array([[2]])
train_x=np.array([[1,2,3,4]])
train_y=np.array([[2],[4],[6],[8]])
#用这个列表装loss数据
LOSS=[]

#初始化权值
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    return w

#正向传播与反向传播
def propagate(w, X, Y):
    #m是输入样本的个数
    m=X.shape[1]
    A=np.dot(X.T,w)
    loss=Y-A
    dw= -1 / m * np.dot(X,(Y-A))
    LOSS.append(loss[0][0])
    return dw

#x,y是输入的数据集，learning_rate是学习速率，num_iterations是学习的次数
def model(x, y, learning_rate, num_iterations):
    w=initialize_with_zeros(x.shape[0])
    for i in range(num_iterations):
        dw = propagate(w, x, y)
        w = w - learning_rate*dw
        print(w)


if __name__ == '__main__':

    # model(train_x, train_y,0.03, 100)
    # #这里是可视化函数，将loss画出来
    # plt.plot(LOSS)
    # plt.show()
    print(np.zeros(1, 3).shape)