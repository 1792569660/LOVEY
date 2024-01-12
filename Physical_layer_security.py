# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# 定义参数
N = 10 # 用户数量
P = 1 # 用户的最大功率
R = 100 # 仿真次数
alpha = np.random.uniform(0.1, 0.5, N) # 功率消耗系数
h = np.random.exponential(1, (N, N)) # 用户之间的信道增益
g = np.random.exponential(0.5, (N, N)) # 用户和攻击者之间的信道增益
sigma2 = 0.01 # 用户的噪声功率
eta2 = 0.01 # 攻击者的噪声功率

# 定义函数
def utility(i, p): # 计算用户i在功率p下的效用值
    u = np.log(1 + h[i, i] * p[i] / (sigma2 + np.sum(h[:, i] * p))) - alpha[i] * p[i]
    return u

def best_response(i, p): # 计算用户i的最佳响应
    p_i = cp.Variable() # 定义优化变量
    objective = cp.Maximize(cp.log(1 + h[i, i] * p_i / (sigma2 + np.sum(h[:, i] * p) - h[i, i] * p[i])) - alpha[i] * p_i) # 定义目标函数
    constraints = [0 <= p_i, p_i <= P] # 定义约束条件
    problem = cp.Problem(objective, constraints) # 定义优化问题
    problem.solve() # 求解优化问题
    br = p_i.value # 获取最优解
    return br

def system_utility(p): # 计算系统的总效用值
    su = np.sum([utility(i, p) for i in range(N)])
    return su

def system_capacity(p): # 计算系统的总信道容量
    sc = np.sum([np.log(1 + h[i, i] * p[i] / (sigma2 + np.sum(h[:, i] * p))) for i in range(N)])
    return sc

def system_power(p): # 计算系统的总功率消耗
    sp = np.sum(p)
    return sp

def attacker_utility(p): # 计算攻击者的效用值
    au = -np.sum([np.log(1 + g[i, i] * p[i] / (eta2 + np.sum(g[:, i] * p))) for i in range(N)])
    return au

def attacker_capacity(p): # 计算攻击者的窃听容量
    ac = np.sum([np.log(1 + g[i, i] * p[i] / (eta2 + np.sum(g[:, i] * p))) - np.log(1 + h[i, i] * p[i] / (sigma2 + np.sum(h[:, i] * p))) for i in range(N)])
    return ac

# 初始化
p = np.random.uniform(0, P, N) # 用户的初始功率
u = np.zeros((N, R)) # 用户的效用值
su = np.zeros(R) # 系统的总效用值
sc = np.zeros(R) # 系统的总信道容量
sp = np.zeros(R) # 系统的总功率消耗
au = np.zeros(R) # 攻击者的效用值
ac = np.zeros(R) # 攻击者的窃听容量

# 仿真
for r in range(R):
    i = np.random.randint(0, N) # 随机选择一个用户
    p[i] = best_response(i, p) # 更新该用户的功率
    u[i, r] = utility(i, p) # 计算该用户的效用值
    su[r] = system_utility(p) # 计算系统的总效用值
    sc[r] = system_capacity(p) # 计算系统的总信道容量
    sp[r] = system_power(p) # 计算系统的总功率消耗
    au[r] = attacker_utility(p) # 计算攻击者的效用值
    ac[r] = attacker_capacity(p) # 计算攻击者的窃听容量

# 绘图
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(p.T)
plt.xlabel('Simulation times')
plt.ylabel('User power')
plt.title('User power choices in physical layer security game')
plt.subplot(2, 3, 2)
plt.plot(u.T)
plt.xlabel('Simulation times')
plt.ylabel('User utility value')
plt.title('User utility values in physical layer security game')
plt.subplot(2, 3, 3)
plt.plot(su)
plt.xlabel('Simulation times')
plt.ylabel('System total utility value')
plt.title('System total utility value in physical layer security game')
plt.subplot(2, 3, 4)
plt.plot(sc)
plt.xlabel('Simulation times')
plt.ylabel('System total capacity')
plt.title('System total capacity in physical layer security game')
plt.subplot(2, 3, 5)
plt.plot(sp)
plt.xlabel('Simulation times')
plt.ylabel('System total power consumption')
plt.title('System total power consumption in physical layer security game')
plt.subplot(2, 3, 6)
plt.plot(au)
plt.xlabel('Simulation times')
plt.ylabel('Attacker utility value')
plt.title('Attacker utility value in physical layer security game')
plt.tight_layout()
plt.show()
