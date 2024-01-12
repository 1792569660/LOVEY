# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
N = 10 # 用户数量
M = 5 # 频段数量
T = 1000 # 仿真时间
R = 100 # 仿真次数
lam = np.random.uniform(0.1, 0.5, N) # 转移强度
beta = np.random.uniform(0.5, 1.5, N) # 风险厌恶系数
c = np.random.uniform(0.01, 0.05, N) # 切换成本
h = np.random.exponential(1, (N, N)) # 信道增益
sigma2 = 0.01 # 噪声功率

# 定义函数
def utility(i, s): # 计算用户i在状态s下的效用值
    u = np.log(1 + h[i, s] / sigma2) - beta[i] * np.sum(h[:, s] / sigma2) - c[i] * np.abs(s - x[i])
    return u

def best_response(i): # 计算用户i的最佳响应
    br = np.argmax([utility(i, s) for s in range(M)])
    return br

def system_utility(): # 计算系统的总效用值
    su = np.sum([utility(i, x[i]) for i in range(N)])
    return su

def system_rate(): # 计算系统的总传输速率
    sr = np.sum([np.log(1 + h[i, x[i]] / sigma2) for i in range(N)])
    return sr

def system_switch(): # 计算系统的总切换次数
    ss = np.sum([np.abs(x[i] - y[i]) for i in range(N)])
    return ss

# 初始化
x = np.random.randint(0, M, N) # 用户的初始状态
y = x.copy() # 用户的上一状态
u = np.zeros((N, T)) # 用户的效用值
su = np.zeros(T) # 系统的总效用值
sr = np.zeros(T) # 系统的总传输速率
ss = np.zeros(T) # 系统的总切换次数

# 仿真
for r in range(R):
    for t in range(T):
        i = np.random.randint(0, N) # 随机选择一个用户
        x[i] = best_response(i) # 更新该用户的状态
        u[i, t] = utility(i, x[i]) # 计算该用户的效用值
        su[t] += system_utility() # 计算系统的总效用值
        sr[t] += system_rate() # 计算系统的总传输速率
        ss[t] += system_switch() # 计算系统的总切换次数
        y = x.copy() # 更新用户的上一状态

# 绘图
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x.T)
plt.xlabel('Time')
plt.ylabel('User state')
plt.title('User state transitions in dynamic spectrum access game')
plt.subplot(2, 2, 2)
plt.plot(u.T)
plt.xlabel('Time')
plt.ylabel('User utility value')
plt.title('User utility values in dynamic spectrum access game')
plt.subplot(2, 2, 3)
plt.plot(sr / R)
plt.xlabel('Time')
plt.ylabel('System total transmission rate')
plt.title('System total transmission rate in dynamic spectrum access game')
plt.subplot(2, 2, 4)
plt.plot(ss / R)
plt.xlabel('Time')
plt.ylabel('System total switching time')
plt.title('System total switching time in dynamic spectrum access game')
plt.tight_layout()
plt.show()
