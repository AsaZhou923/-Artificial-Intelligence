#基本库函数导入
import numpy as np
import os
#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
#模拟数据生成
import numpy as np
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)
plt.plot(X,y,'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([0,2,0,15])
plt.show()
#最小二乘法代码实现
X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#array([[4.21509616],[2.77011339]])
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_best)
y_predict
plt.plot(X_new,y_predict,'r--')
plt.plot(X,y,'b.')
plt.axis([0,2,0,15])
plt.show()
#批量梯度下降代码过程
eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
for iteration in range(n_iterations):
    gradients = 2/m* X_b.T.dot(X_b.dot(theta) - y)
theta = theta - eta*gradients
#学习率对梯度下降结果的影响
theta_path_bgd = []
def plot_gradient_descent(theta,eta,theta_path = None):
    m = len(X_b)
    plt.plot(X,y,'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new,y_predict,'b-')
        gradients = 2/m* X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('X_1')
    plt.axis([0,2,0,15])
plt.title('eta = {}'.format(eta))

theta = np.random.randn(2,1)
plt.figure(figsize=(10,4))
plt.subplot(131)
plot_gradient_descent(theta,eta=0.02)
plt.subplot(132)
plot_gradient_descent(theta,eta=0.1,theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta,eta=0.5)
plt.show()
#随机梯度下降代码实现
theta_path_sgd=[]
m = len(X_b)
np.random.seed(42)
n_epochs = 50

t0 = 5
t1 = 50

def learning_schedule(t):
    return t0/(t1+t)    #定义变化的学习率

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch < 10 and i < 10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new,y_predict,'r-')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1] #随机从数据集中取数据
        gradients = 2* xi.T.dot(xi.dot(theta)-yi)
#根据随机梯度下降公式计算梯度变化值
        eta = learning_schedule(epoch*m+1)  #调用函数计算学习率
        theta = theta-eta*gradients  #计算参数值
        theta_path_sgd.append(theta)

plt.plot(X,y,'b.')
plt.axis([0,2,0,15])
plt.show()

#小批量梯度下降代码实现
theta_path_mgd=[]
n_epochs = 50
minibatch = 16
theta = np.random.randn(2,1)
t0,t1 = 200,1000
def learning_schedule(t):
    return t0 / (t + t1)
np.random.seed(42)
t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]# 将数据打乱
    for i in range(0, m, minibatch):
        t += 1
        xi = X_b_shuffled[i:i + minibatch]
        yi = y_shuffled[i: i + minibatch]
        gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)
        # 使用小批量梯度下降公式计算梯度变化值
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

#对比不同梯度下降策略
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)
#上面3行代码分别计算不同梯度下降方法中的参数变化值
plt.figure(figsize=(12,6))
plt. plot(theta_path_sgd[:,0], theta_path_sgd[:,1], 'r-s',linewidth=1,label='SGD')
plt. plot(theta_path_mgd[:,0], theta_path_mgd[:,1], 'g-+',linewidth=2,label='MINIGD')
plt. plot(theta_path_bgd[:,0], theta_path_bgd[:,1], 'b-o',linewidth=3,label='BGD')
# 分别绘制3种梯度下降方法的参数变化曲线
plt.legend(loc= 'upper left')
plt. axis([3.5,4.5,2.0,4.0])
plt. show()