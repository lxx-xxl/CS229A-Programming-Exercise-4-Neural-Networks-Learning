# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:02:38 2021

@author: taylo
"""
#https://blog.csdn.net/m0_37867091/article/details/104988507
import os
import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy import random

#<exclude bias units>400 units in input layer, 25 units in hidden layer (only 1 hidden), 10 units in output units
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\04-neural_network(bp)')
data = scio.loadmat('ex4data1.mat') #读取出来的data是字典格式
print (len(data['X']))
raw_X = data['X']	# raw_X 维度是(5000,400)
raw_y = data['y']	# raw_y 维度是(5000,1)

#visualizing the data
def plot_image(X):
    # 创建绘图实例
        #100×50共5000张图，总窗口大小为2×2,#所有图像共享x,y轴属性, 这个是先整一个框出来, 然后再填pixel进去
    fig,ax = plt.subplots(ncols=100,nrows=50,figsize=(2,2),sharex=True,sharey=True)
            #隐藏x,y轴坐标刻度
    plt.xticks([])
    plt.yticks([])
    
    for a in range(5):
        for b in range(10):
            for c in range(10):
                for d in range(10):
                    ax[10*a+d,10*c+d].imshow(X[1000*a+100*b+10*c+d].reshape(20,20).T,cmap='gray_r')
    plt.show()
#plot_image(raw_X)

#Forward prop to compute activation for all examples and all layers
#insert bias units
x = np.insert(raw_X, 0, 1, axis = 1) #(5000,401)
X = np.matrix(x) # X维度是(5000,401)
print(X.shape)
#y=raw_y
y = np.matrix(raw_y) # raw_y 维度是(5000,1)
print(y.shape)
#reshape Y into multi-classification expression(就是把本顺序令为1，别的令为0)
#0放在最后
def one_hot_encoder(y):
    result=[]
    for i in y:
        y_temp=np.zeros(10)
        y_temp[i-1]=1
        result.append(y_temp)
    return np.array(result)

Y = one_hot_encoder(y)  #5000*10

#define theta
raw_theta = scio.loadmat('ex4weights.mat')
theta_1=raw_theta['Theta1'] # theta1维度是(25*401)
theta_2=raw_theta['Theta2'] # theta2维度是(10*26)
    #unroll theta把俩theta合并转化为一维数组
#将theta1&2转化为一维数组
def serialize(a,b):
#    print('check'+str(a.flatten().shape)+str(b.flatten().shape))
    return np.hstack((a.flatten(),b.flatten()))
theta_serialize=serialize(theta_1,theta_2)#.reshape(1,10285)#.reshape(2057,5).flatten()
#print(theta_serialize[:25*401].shape) #只能同纬度reshape，切片没问题
def deserialize(theta_serialize):
    theta1 = np.array(theta_serialize[:25*401]).reshape(25,401)
    theta2 = (theta_serialize[25*401:].reshape(1,260)).reshape(10,26)
    return theta1, theta2


def sigmoid(z): 
    return 1/(1+np.exp(-z))
#compute activations
def forward_prop(theta_serialize, X):
#    theta1, theta2=deserialize(theta_serialize)
    a1=X
    z2=a1*theta_1.T     #(5000,401)*(401,25)=(5000,25)
    a2=sigmoid(z2)
    a2=np.insert(a2,0,1,axis=1) #(5000,26)
    z3=a2*theta_2.T     #(5000,26)*(26,10)=(5000,10)
    h = sigmoid(z3)    #(5000,10)
    return a1,z2,a2,z3,h

def costfunction_notreg(theta_serialize, X, Y):
    a1,z2,a2,z3,h = forward_prop(theta_serialize, X)
    y_1=np.multiply(Y,np.log(h))                        
    y_2=np.multiply((1-Y),np.log(1-h))                  
#    reg = np.dot(np.sum(np.power(theta[1:], 2)), lamda/2*m)
    return np.sum(-y_1-y_2)/len(X)      #(1*1)

costfunc_notreg = costfunction_notreg(theta_serialize, X, Y)
print(costfunc_notreg)  #答案对的
#if costfunc_notreg == 0.287629:
#    print('True')
#else:
#    print('Fause')

def costfunction_reg(theta_serialize, X, Y, lamda):
    a1,z2,a2,z3,h = forward_prop(theta_serialize,X)
#    theta1, theta2=deserialize(theta_serialize)
    y_1=np.multiply(Y,np.log(h))                        
    y_2=np.multiply((1-Y),np.log(1-h))  
    sum_theta1 = np.sum(np.power(theta_1[:, 1:],2))  #(1*1)
    sum_theta2 = np.sum(np.power(theta_2[:, 1:],2))  #(1*1)
    reg = np.dot(np.sum(sum_theta1+sum_theta2),lamda/(2*len(X))) #(1*1)
    return np.sum(-y_1-y_2)/len(X)+reg

lamda = 1
costfunc_reg = costfunction_reg(theta_serialize, X, Y, lamda)
print(costfunc_reg) #对的
#if costfunc_reg == 0.383770:
#    print('True')
#else:
#    print('Fause')

#Back prop
#sigmoid函数求导
def sigmoidgradientfunc(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def gradient_unreg(theta_serialize,X, Y):
#    theta1, theta2=deserialize(theta_serialize)
    a1,z2,a2,z3,h = forward_prop(theta_serialize,X)
    d3 = h-Y     #5000*10
    d2 = np.multiply(d3*theta_2[:,1:],sigmoidgradientfunc(z2))   #(5000*25).*(5000*25)=(5000*25) 
    D_2=a2.T*d3/len(X)     #(26*10)
    D_1=a1.T*d2/len(X)     #(401*25)
#    print('D2: ' + str(D_2.flatten().shape)+'D1:'+str(D_1.flatten().shape))
    return serialize(D_1, D_2)

def gradient_reg(theta_serialize, X, Y, lamda):
#    theta1, theta2 = deserialize(theta_serialize)
    g_reg1_temp=lamda*theta_1[:, 1:]     #(25,400)
    g_reg2_temp=lamda*theta_2[:, 1:]     #(10,25)
    g_reg1 = np.insert(g_reg1_temp, 0, 0, axis = 1) #(25,401)
    g_reg2 = np.insert(g_reg2_temp, 0, 0, axis = 1) #(10*26)
    g_reg = serialize(g_reg1,g_reg2)
    g_unreg=gradient_unreg(theta_serialize, X, Y)
    return g_reg+g_unreg
    
def randominitialization():
    return random.uniform(-0.12,0.12,10285)


def gradient_checking(epsilon):
    #think of theta1&2 as a long vector
    e_vec = np.zeros(10285) #(10285,1)
    overall = []
    for i in range(0,10285):  #i=[0,10284]
        e_vec[i] = epsilon
        gradcheck=((theta_serialize+np.array(e_vec))-(theta_serialize-np.array(e_vec)))/(2*epsilon)
        overall.append(gradcheck)
    #假设相比小于几就算是similar：
    if np.sum(overall) - costfunc_reg < 0.01:
        print('gradient checking pass:'+str(np.sum(overall) - costfunc_reg))
    else:
        print('gradient checking fail')
    print(str(np.sum(overall)) +':'+ str(costfunc_reg))
epsilon = 0.0001
check = gradient_checking(epsilon)
#定义优化函数（抄的）
def training(X, Y, lamda):
    thetarandomini = randominitialization()
    res = minimize(fun = costfunction_reg, x0=thetarandomini, args = (X, Y, lamda), method = 'TNC', jac = gradient_reg, options = {'maxiter':300})
    return res
#计算网络准确率(抄的)
lamda = 10
res = training(X,Y, lamda)
_,_,_,_,h = forward_prop(theta_serialize, X)

y_pred = np.argmax(h,axis=1)+1  #10个里返回值最大的 (5000,1)
acc = np.mean(y_pred == y)
print(acc)

#visualize hidden layer
def plot_hiddenlayer(hidden):
    # 创建绘图实例
        #5×5共25张图，总窗口大小为4×4,#所有图像共享x,y轴属性, 这个是先整一个框出来, 然后再填pixel进去
    fig,ax = plt.subplots(ncols=5,nrows=5,figsize=(5,5),sharex=True,sharey=True)
    #隐藏x,y轴坐标刻度
    plt.xticks([])
    plt.yticks([])
    for i in range(5):  #[0,1,2,3,4]
        for j in range(5):  #hidden索引[0:24]
            ax[i,j].imshow(hidden[5*i+j].reshape(20,20).T,cmap='gray_r')
    plt.show()
hidden = theta_1[:,1:]
hiddenfig = plot_hiddenlayer(hidden)