# AI社会 经济学第二次作业

## 题目1 复现劳动不可分的RBC模型

本次试验主要是复现一个不可分劳动的模型，即要么全职工作，要么失业（即Notebook中没有覆盖到的模型B）。

延续Notebook中的架构，分以下几个步骤进行实现——

1. 使用Tauchen方法把环境离散化：把技术冲击（TFP）设定为一个 AR(1) 过程，使用 `approx_markov` 函数（Tauchen 方法）把连续的冲击变成了 7 个离散的状态，生成了转移矩阵。

2. 使用值函数迭代的方法求解模型：使用网格离散化，资本存量$k$​具有50个grid而冲击具有7个状态。

   在代码的 `utility` 函数中，对两种模型设定了不同的效用形式。

   - 模型 B 使用标准的对数效用函数：$U = \log(c) + A \log(1-l)$。
   - 模型 A 在宏观加总层面表现为线性效用函数：$U = \log(c) + B(1-l)$。

3. 模拟：值函数收敛之后，模拟2000期经济数据，对模拟出的数据取对数，使用 **HP 滤波**（参数 $\lambda=1600$）提取周期项，最后算标准差和相关系数。

实现结果如下

```
====== 模型 A: 劳动不可分 (Indivisible Labor) ======
   Variable  Std Dev (%)  Corr w/ Output
     Output     2.200542        1.000000
Consumption     0.822214        0.453017
 Investment     8.014043        0.957175
    Capital     0.627188        0.068133
      Labor     1.969548        0.928165
        TFP     1.089942        0.931423
Relative Volatility (Labor/Output): 0.90

====== 模型 B: 劳动可分 (Divisible Labor) ======
   Variable  Std Dev (%)  Corr w/ Output
     Output     1.634935        1.000000
Consumption     0.909716        0.280481
 Investment     6.281039        0.908522
    Capital     0.523859        0.073834
      Labor     1.094975        0.844858
        TFP     1.089942        0.944039
Relative Volatility (Labor/Output): 0.67
```

**经济学逻辑解释**

相比于标准模型假设人们仅仅调整自己的工作时长，Hanssen模型引入了要么工作，要么不工作的约束。在劳动可分模型中，由于收入的边际效用递减，劳动的负效用递增的规律，人们倾向于平滑自己的闲暇时间，不愿意让工作时长剧烈波动。

经查阅资料和课堂讲解发现，Hanssen 利用 Rogerson (1988) 的彩票均衡理论证明，虽然个体的效用函数是凹的，但在宏观加总层面，代表性家庭的效用函数对劳动变成了**线性关系** (Linear in Labor)，边际效用递减的效应消失了。于是这样的线性关系带来了非常大的“跨期替代弹性”，工资微小的上涨就能带来劳动供给的大幅增加，大批人从失业变成就业，劳动供给对工资变化变得很敏感，劳动波动幅度大幅提升，导致Relative Volatility远高。

## 题目2 复现Aiyagari94模型

复现Aiyagari在1994年提出的具有异质性主体和不完全市场的一般均衡模型。复现利率、资本存量、财富和消费基尼。

参考PPT中给出的实现逻辑，我们分一下几个部分进行复现：

1. 离散化：采用修正后（扩大了网格边界）的 Tauchen 方法将连续的 AR(1) 收入过程离散化为马尔可夫链。

2. 值函数迭代

   利用 NumPy 的broadcast进行全矩阵运算，通过 `cons_mat = (1 + r) * ka_grid + w * s_grid_re - kap_grid` 一次性计算出所有可能状态组合下的消费矩阵。

   采用**直方图累加法**等价实现了稀疏矩阵乘法。利用 `np.add.at` 函数，根据最优策略索引 `policy_idx`，直接将当前期的概率质量 (Mass) “搬运”到下一期的对应位置。

3. 均衡求解：求均衡条件$K_s=K_d$下各个参量的值。使用二分法搜索均衡利率$r$，根据资料和理论估计，将初始区间设定为 $[0.001, 1/\beta - 1]$。

复现结果

```
========================================
1. 均衡利率 (r)       : 1.5177% (Target: ~1.28%)
2. 资本存量 (K)       : 7.9947
3. 储蓄率 (Inv/Y)     : 30.26% (Target: ~31%)
4. 财富基尼 (Gini_W)  : 0.4878
5. 消费基尼 (Gini_C)  : 0.1726
========================================
(Target为原论文中给出的模拟结果)
```

均衡利率与Aiyagari94的原论文存在一定差异，查询资料发现可能是由于数值方法带来的$\sigma$差异。

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/经济学模型复现作业/Figure_1.png" alt="Figure_1" style="zoom:72%;" />

## 题目3 利用深度学习方法实现K&S模型求解

模型搭建：

经济中存在无限多的家庭和代表性企业。

1. 家庭问题：

   家庭在每一期通过选择消费 ($c$) 和下一期资产 ($a'$) 来最大化终身期望效用：

   $$V(z, a, \Gamma) = \max_{c, a'} \{ u(c) + \beta E [V(z', a', \Gamma') | z, \Gamma] \}$$

   约束条件为：

   - **预算约束：** $c + a' = (1+r)a + w l z$

   - 借贷约束： $a' \ge 0$

     其中 $\Gamma$ 是当前的资产分布，$z$ 是个体生产率冲击。

2. 企业问题：

   代表性企业利用资本 ($K$) 和劳动 ($L$) 进行生产：

   $$Y = \mathrm{TFP} \cdot K^\alpha L^{1-\alpha}$$

   企业根据边际产出决定工资 $w$ 和利率 $r$。

3. 市场出清：

   总资本 $K$ 等于所有家庭资产 $a$ 的积分（加总）。

在本次模拟中，使用了和HA_model中一致的参数，折现因子 $\beta = 0.895$，风险厌恶系数 $\sigma = 2.0$，资本份额 $\alpha = 0.34$，折旧率 $\delta = 0.06$

主要实现了`module_obj_euler`和`module_training_euler`两个与最小化欧拉残差训练相关的逻辑，直接寻找满足一阶最优性条件的策略函数，而不需要显式求解价值函数。

具体求解过程中，使用了PPT中提供的Fischer-Burmeister (FB) 函数处理约束

$$\Psi(a, b) = a + b - \sqrt{a^2 + b^2}$$，该函数具有性质：$\Psi(a, b) = 0 \iff a \ge 0, b \ge 0, ab=0$。

训练参数：神经网络中，`n1_p, n2_p`设置为24，batchsize为128，使用Adam优化器，基础学习率设置为0.1，蒙特卡洛采样数为50，训练epoch为250

目前仍然存在一定问题（少数情况下，值函数和策略函数图像会出现形状上的异常），但绝大部分时候能较为准确地预测宏观资产分布。

模拟后结果如下

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/吕鸿坤/EulerResidual/figures/scatter_policy_a1.png" alt="scatter_policy_a1" style="zoom:50%;" />

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/吕鸿坤/EulerResidual/figures/scatter_policy_c.png" alt="scatter_policy_c" style="zoom:50%;" />

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/吕鸿坤/EulerResidual/figures/scatter_value.png" alt="scatter_value" style="zoom:50%;" />

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/吕鸿坤/EulerResidual/figures/sim_path_2d.png" alt="sim_path_2d" style="zoom:50%;" />

<img src="/Users/lyuhongkun/Desktop/PKU/课程文件/大二上/AI社会/经济学/吕鸿坤/EulerResidual/figures/sim_path_aggregates.png" alt="sim_path_aggregates" style="zoom:50%;" />

`Duration: 0:00:31.341627`。

欧拉方程（一阶最优性条件 FOC）的核心思想是**“当前消费的边际效用必须等于未来消费的期望贴现边际效用”**。由于欧拉残差法是局部优化方法，只需要在当前状态下验证 FOC 是否成立。计算损失时，它只需要**一次**（或两次，用于方差降低的“双重采样”）**下一期的预测和期望**，而不需要进行 50 步的完整路径模拟。它绕过了显式地拟合复杂的“价值函数”（即贝尔曼方法中的 $V$），而是直接将目标设定为最小化 FOC 的残差，且MC模拟样本数目也比较少，从而每个epoch训练时间比较短。