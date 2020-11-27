#linear fitting.py
#生成服从一维正态分布的随机数（离散值），使用最小二乘法进行曲线拟合，并梯度下降法求取极值。
#导入2d图形库 matplotlib 数学函数库numpy mah
import numpy as np
import matplotlib.pyplot as plt
import math

#生成20个待测试的服从标准正态分布随机数并且打印在图像上


X = np.arange(-5, 5, 0.1)
Z = [1/math.sqrt(2*math.pi)*math.exp(-x**2/2) for x in X]
Y = np.array([np.random.normal(z,0.9) for z in Z])
plt.plot(X,Y,'ro')
plt.show()


"""
           1
    f(x)= −−exp(−(x−μ)2/2σ2)
         √2πσ
"""

# 求解多项式的正则方程组
# 生成系数矩阵A
def gen_coefficient_matrix(X, Y):
    N = len(X)
    m = 9
    A = []
    # 计算每一个方程的系数
    for i in range(m):
        a = []
        # 计算当前方程中的每一个系数
        for j in range(m):
            a.append(sum(X ** (i+j)))
        A.append(a)
    return A

# 计算方程组的右端向量b
def gen_right_vector(X, Y):
    N = len(X)
    m = 9
    b = []
    for i in range(m):
        b.append(sum(X**i * Y))
    return b

A = gen_coefficient_matrix(X, Y)
b = gen_right_vector(X, Y)

a0, a1, a2 ,a3,a4,a5,a6,a7,a8= np.linalg.solve(A, b)  

# 生成拟合曲线的绘制点
_X = np.arange(-5, 5, 0.01)
_Y = np.array([a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6+a7*x**7 +a8*x**8 for x in _X])

plt.plot( _X, _Y, 'b', linewidth=2)

plt.show()

'''
def dj(theta):
	return (a1+2*a2*theta+3*a3*theta**2+4*a4*theta**3+5*a5*theta**4+6*a6*theta**5)#这里返回theta对应的导数，而函数求导是手动进行的
def j(theta):
	return (a0 + a1*theta + a2*theta**2 + a3*theta**3 + a4*theta**4 + a5*theta**5 + a6*theta**6  )
    
eta=0.1#这个是学习率
theta=0.0#第一个点
episilon=1e-8
while True:
	gradient=dj(theta)
	last_theta=theta
	theta-=gradient*eta#对theta进行更新
	if abs(j(theta)-j(last_theta)<episilon):#如果小于一个阈值，那么就退出
		break
print(theta)
print(j(theta))

'''
