## Chap7.参数估计

对总体中的未知参数给出估计

1. 点估计，利用统计量和样本给出一个近似值
2. 区间估计，给出真实值可能落入的范围，并告诉可信程度有多高



### 53.点估计与矩估计

**点估计问题：**

设总体 $X$ 的分布函数（概率密度，分布律）的形式已知，但它的一个或多个参数未知，借助于总体 $X$ 的一个样本来估计总体未知参数的值的问题，称为参数的点估计问题。

> 具体来说，比如已经知道一组数据已知服从泊松分布，但是具体的参数 $\lambda$ 未知。



点估计问题的提法：

- **已知**：总体 $X$ 的分布函数的 $F(x;\theta)$ 的形式，
- **未知**：$\theta$ 待估参数。$\theta = (\theta_1, \theta_2, \theta_3)$
- **利用**：$X_1, X_2, \dots, X_n$ 是 $X$ 的一个样本，$x_1, x_2, \dots, x_n$ 是相应一个样本值。



**点估计的解决办法：**

构造一个适当的统计量 $\hat{\theta}(X_1, X_2, \dots, X_n)$，用它的观察值 $\hat{\theta}(x_1, x_2, \dots, x_n)$ 作为未知参数 $\theta$ 的近似值。
- 称 $\hat{\theta}(X_1, X_2, \dots, X_n)$ 为 $\theta$ 的估计量。
- 称 $\hat{\theta}(x_1, x_2, \dots, x_n)$ 为 $\theta$ 的估计值。

注：

1. 称估计量和估计值为估计。
2. 由于估计量是样本的函数，对不同样本值，估计值一般不同。



#### 矩估计

理论依据是：样本的 $k$ 阶矩依概率收敛于总体的 $k$ 阶矩



总体 $X$ 的 $k$ 阶原点矩 $E[X^k]$

样本 $x_1,x_2,...,x_n$ 的 $k$ 阶原点矩 $A_k=\frac{1}{n}\sum_{i=1}^{n}x_i^k$

**借助样本的 $k$ 阶原点矩近似估计总体 $X$ 的 $k$ 阶原点矩**



**矩估计求解步骤**

设总体 $X$ 的分布中含有 $m$ 个未知参数 $\theta_1, \theta_2, \dots, \theta_m$，则：

1. 求总体的各阶矩 $E(X^k) \quad (k=1, 2, \dots, m)$；
2. 令样本的各阶矩等于总体的各阶矩，得到含有 $m$ 个未知参数 $\theta_1, \theta_2, \dots, \theta_m$ 的方程；

$$
\frac{1}{n} \sum_{i=1}^{n} X_i = E(X)
$$
$$
\frac{1}{n} \sum_{i=1}^{n} X_i^2 = E(X^2)
$$
$$
\vdots
$$
$$
\frac{1}{n} \sum_{i=1}^{n} X_i^m = E(X^m)
$$

3. 解上述方程，所求得的解 $\hat{\theta}_k(X_1, X_2, \dots, X_n)$ 称为未知参数 $\theta_k$ 的估计量，简称估计。

> 实际上，几个参数就写到几阶矩，例如正态分布两个参数，就写到两阶矩

常用的一个公式：
$$
E(X^2) = D(X) + E^2(X)
$$




设总体 $X$ 的均值 $\mu$ 及方差 $\sigma^2$ 都存在，且 $\sigma^2 > 0$。但 $\mu$，$\sigma^2$ 均为未知。又设 $X_1, X_2, \dots, X_n$ 是来自 $X$ 的样本，试求 $\mu$，$\sigma^2$ 的估计量。
$$
E(x) = \mu\\
E(x) = D(x) + E^2(x) = \sigma^2 + \mu^2\\
A_1 = \overline{X} = \mu\\
A_2 = \frac{1}{n} \sum_{i=1}^{n} X_i^2 = \sigma^2 + \mu^2
$$
则有  
$$
\mu = \overline{X}
$$

$$
\sigma^2 + \mu^2 = \frac{1}{n} \sum_{i=1}^{n} X_i^2
$$

$$
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} X_i^2 - \overline{X}^2 = \frac{1}{n} \sum_{i=1}^{n} (X_i - \overline{X})^2
$$

也就是说，不管 $X$ 的分布是什么，总体的均值与方差的估计量是相同的。











### 54.最大似然估计

#### 离散型

设总体 $X$ 为离散型，分布律已知，但分布中含有 $m$ 个未知参数 $\theta_1, \theta_2, \dots, \theta_m$，$X_1, X_2, \dots, X_n$ 是 $X$ 的一个样本，$x_1, x_2, \dots, x_n$ 是相应的样本值。

易知 $X_1, X_2, \dots, X_n$ 取到观察值 $x_1, x_2, \dots, x_n$ 的概率，即事件 $P\{X_1 = x_1, X_2 = x_2, \dots, X_n = x_n\}$ 发生的概率为

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \prod_{i=1}^{n} P\{X = x_i\}
$$

这一步骤的概率随 $\theta_1, \theta_2, \dots, \theta_m$ 的取值而变化，它是 $m$ 个未知参数 $\theta_1, \theta_2, \dots, \theta_m$ 的函数，称其为样本的似然函数，需注意的是 $x_1, x_2, \dots, x_n$ 是已知的样本值。



**最大似然估计法的思想：**

已知样本值 $x_1, x_2, \dots, x_n$ 了，这时说取到这一样本值的概率函数
$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)
$$

比较大。因此，可以固定样本观察值 $x_1, x_2, \dots, x_n$，挑选使得似然函数

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)
$$

达到最大值的参数值 $\hat{\theta}_i(x_1, x_2, \dots, x_n) \quad (i = 1, 2, \dots, m)$。



用这思想求出的参数值
$$
\hat{\theta}_i(x_1, x_2, \dots, x_n) \quad (i = 1, 2, \dots, m)
$$

称为 $\theta_1, \theta_2, \dots, \theta_m$ 的最大似然估计量。



相应的统计量 $\hat{\theta}_i(X_1, X_2, \dots, X_n), \quad i = 1, 2, \dots, m$ 称为参数的最大似然估计量。





#### 连续型

若总体 $X$ 是连续型，$X_1, X_2, \dots, X_n$ 来自总体 $X$ 的一个样本，$x_1, x_2, \dots, x_n$ 是相应的一个样本值，则 $X_1, X_2, \dots, X_n$ 的联合概率密度函数为

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \prod_{i=1}^{n} f(x_i; \theta_1, \theta_2, \dots, \theta_m)
$$

随机点 $(X_1, X_2, \dots, X_n)$ 落在点 $(x_1, x_2, \dots, x_n)$ 的邻域（边长分别为 $dx_1, dx_2, \dots, dx_n$ 的 $n$ 为立方体）内的概率近似为

$$
\prod_{i=1}^{n} f(x_i; \theta_1, \theta_2, \dots, \theta_m) dx_i
$$

其值随 $\theta_1, \theta_2, \dots, \theta_m$ 的取值而变化，最大似然估计的思想是取 $\theta_1, \theta_2, \dots, \theta_m$ 的估计值使得上述概率取得最大值。注意到 $\prod_{i=1}^{n} dx_i$ 不随 $\theta_1, \theta_2, \dots, \theta_m$ 而变，因此只需考虑函数

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)
$$

的最大值。称为样本的似然函数。

使似然函数取得最大值的参数值

$$
\hat{\theta}_i(x_1, x_2, \dots, x_n) \quad (i = 1, 2, \dots, m)
$$

称为 $\theta_1, \theta_2, \dots, \theta_m$ 的最大似然估计值。

相应的统计量 $\hat{\theta}_i(X_1, X_2, \dots, X_n), \quad i = 1, 2, \dots, m$ 称为参数的最大似然估计量。



#### 求解步骤

(1) 写出似然函数

若 $X$ 为离散型，似然函数为

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \prod_{i=1}^{n} P\{X = x_i\}
$$

若 $X$ 为连续型，似然函数为

$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \prod_{i=1}^{n} f(x_i; \theta_1, \theta_2, \dots, \theta_m)
$$

(2) 对似然函数两边取对数，得到对数似然函数
$$
\ln L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \sum_{i=1}^{n} \ln P\{X = x_i\}
$$

或

$$
\ln L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m) = \sum_{i=1}^{n} \ln f(x_i; \theta_1, \theta_2, \dots, \theta_m)
$$

(3) 对数似然函数关于各未知参数求偏导，得到对数似然方程
$$
\frac{\partial \ln L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)}{\partial \theta_1} = 0
$$

$$
\frac{\partial \ln L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)}{\partial \theta_2} = 0
$$

$$
\vdots
$$

$$
\frac{\partial \ln L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)}{\partial \theta_m} = 0
$$

(4) 求解对数似然方程，若有解
$$
\theta_1 = \theta_1(X_1, X_2, \dots, X_n)
$$

$$
\theta_2 = \theta_2(X_1, X_2, \dots, X_n)
$$

$$
\vdots
$$

$$
\theta_m = \theta_m(X_1, X_2, \dots, X_n)
$$

则是 $\theta_1, \theta_2, \dots, \theta_m$ 的最大似然估计量。

(5) 当对数似然方程无解时，利用高等数学中的单调性直接观察似然函数
$$
L(x_1, x_2, \dots, x_n; \theta_1, \theta_2, \dots, \theta_m)
$$

达到最大值时的 $\theta_i(x_1, x_2, \dots, x_n)$ 即可。



#### exp1：

1. 设 $X \sim b(1, p)$，$X_1, X_2, \dots, X_n$ 来自 $X$ 的一个样本，试求参数 $p$ 的最大似然估计量。

解：

设 $X_1, X_2, \dots, X_n$ 为独立的样本，且每个样本服从 $X \sim b(1, p)$ 的二项分布。则

$$
P\{X_i = x_i\} = p^{x_i} (1-p)^{1-x_i}, \quad x_i = 0, 1
$$

于是样本的联合分布为

$$
L(p) = P\{X_1 = x_1, X_2 = x_2, \dots, X_n = x_n\}
$$

$$
= \prod_{i=1}^{n} P\{X_i = x_i\}
$$

$$
= \prod_{i=1}^{n} p^{x_i} (1-p)^{1-x_i}
$$

$$
= p^{\sum x_i} (1-p)^{n - \sum x_i}
$$

对似然函数取对数
$$
\ln L(p) = \ln \left[ p^{\sum x_i} (1-p)^{n - \sum x_i} \right]
$$

$$
= \sum x_i \ln p + (n - \sum x_i) \ln (1-p)
$$

求导数，得对数似然方程
$$
\frac{d \ln L(p)}{dp} = \frac{\sum x_i}{p} - \frac{n - \sum x_i}{1 - p}
$$

令其为零，得到

$$
\frac{\sum x_i}{p} = \frac{n - \sum x_i}{1 - p}
$$

解方程，得到 $p$ 的最大似然估计值
$$
\hat{p} = \frac{\sum x_i}{n} = \overline{X}
$$


#### exp2：

![截屏2025-02-22 22.27.57](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-22 22.27.57.png)

设 $X \sim N(\mu, \sigma^2)$，则其概率密度函数为

$$
f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}, \quad x \in \mathbb{R}
$$

样本的似然函数为

$$
L(\mu, \sigma^2) = \prod_{i=1}^{n} f(x_i) = \left(\frac{1}{\sqrt{2 \pi \sigma^2}}\right)^n e^{-\frac{1}{2 \sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2}
$$

对似然函数取对数：
$$
\ln L(\mu, \sigma^2) = -\frac{n}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2
$$

对 $\mu$ 和 $\sigma^2$ 求偏导，得到对数似然方程：
$$
\frac{\partial \ln L(\mu, \sigma^2)}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i - \mu) = 0
$$

$$
\frac{\partial \ln L(\mu, \sigma^2)}{\partial \sigma^2} = -\frac{n}{2 \sigma^2} + \frac{1}{2 (\sigma^2)^2} \sum_{i=1}^{n} (x_i - \mu)^2 = 0
$$

样本的均值和方差的最大似然估计量：

$$
\hat{\mu} = \overline{x}, \quad \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \overline{x})^2
$$



#### exp3：

![截屏2025-02-22 22.27.29](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-22 22.27.29.png)

解：样本：$X_1, X_2, \dots, X_n$，$X_i$ 的概率密度为
$$
f(x) = \begin{cases} 
\frac{1}{b-a}, & a \leq x \leq b \\
0, & \text{其他}
\end{cases}
$$

似然函数：
$$
L(a, b) = \prod_{i=1}^{n} f(x_i) = \left(\frac{1}{b-a}\right)^n, \quad a \leq x_i \leq b
$$

$$
\ln L(a, b) = n \ln \left(\frac{1}{b-a}\right) = -n \ln(b-a)
$$

求导数：
$$
\frac{\partial \ln L(a, b)}{\partial a} = -\frac{n}{b-a}
$$

$$
\frac{\partial \ln L(a, b)}{\partial b} = -\frac{n}{b-a}
$$

**求解上面方程：无解**





问题：最大似然估计的思想：

设 $a, b$ 满足 $a \leq x_1 \leq x_2 \leq \dots \leq x_n \leq b$ 时

$$
L(a, b) = \frac{1}{(b-a)^n}
$$

得到最大值的参数值：

- 分析：$b - a$ 越小，$L(a, b)$ 值越大；
- 当 $b$ 越小，$a$ 越大时，$b - a$ 越小。



当 $a \leq x_1 \leq x_2 \leq \dots \leq x_n \leq b$ 时，设 $\hat{a} = \min \{x_1, x_2, \dots, x_n\}$；设 $\hat{b} = \max \{x_1, x_2, \dots, x_n\}$。

$$
\hat{a} = \min \{X_1, X_2, \dots, X_n\}
$$

$$
\hat{b} = \max \{X_1, X_2, \dots, X_n\}
$$



#### exp4：

![截屏2025-02-22 22.34.04](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-22 22.34.04.png)

$E(X) =3-4\theta$

 $\overline{X} = E(X) = 3 -4\theta = 2$  ，$\hat{\theta} = \frac{3}{4}$ 是求得的 $\theta$ 的最大似然估计量。


$$
L(\theta)=P\{X_1=3,X_2=1,......X_8=3\}=4\theta^6(1-\theta)^2(1-2\theta)^4
$$

$$
\ln L(\theta) = \ln 4+6\ln\theta + 2\ln (1 - \theta) + 4\ln (2 - \theta)
$$



### 55.估计量的评选标准

矩估计和最大似然估计，对同一组样本的估计值可能不同？

怎么评估估计量的好坏呢？



估计量是随机变量，对于不同的样本值，有不同的估计值，希望这些估计只最好在待估参数真值的附近，判断这种性质的有效方法是：无偏性。



#### 无偏性

**定义：**设 $\hat{\theta} = \theta(X_1, X_2, \dots, X_n)$ 是总体参数 $\theta$ 的估计量。如果估计量 $\hat{\theta}$ 的期望值 $E(\hat{\theta}) = \theta$，则我们称 $\hat{\theta}$ 为 $\theta$ 的无偏估计。



样本方差：
$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar{X})^2
$$
因为 $\frac{1}{n}$ 不是总体 $\sigma^ 2$ 的无偏估计。



**样本的 $k$ 阶矩 $A_k$ 是总体 $k$ 阶矩 $\mu_k$ 的无偏估计**



#### 有效性

设 $\hat{\theta_1}$ 和 $\hat{\theta_2}$ 均为 $\theta$ 的无偏估计。如果有：$D(\hat{\theta_1}) \leq D(\hat{\theta_2})$，则称 $\hat{\theta_1}$ 比 $\hat{\theta_2}$ 更有效。



#### 相合性

设 $\hat{\theta} = \theta(X_1, X_2, \dots, X_n)$ 是总体参数 $\theta$ 的估计量。如果对任意的 $\epsilon > 0$，有：
$$
\lim_{n \to \infty} P\left( |\hat{\theta} - \theta| < \epsilon \right) = 1
$$
即随着样本量 $n$ 趋近于无穷大，估计量 $\hat{\theta}$ 收敛于真实参数 $\theta$，则称 $\hat{\theta} = \theta(X_1, X_2, \dots, X_n)$ 是 $\theta$ 的相合估计量，或者又叫做一致估计量。



**样本的 $k$ 阶矩 依概率收敛与总体的 $k$ 阶矩，所以样本的 $k$ 阶矩是总体 $k$ 阶矩的相合估计量。**



**样本均值 $\bar{X}$ 是总体均值 $E[X]$ 的相合估计量**

**样本方差 $s^2$ 是总体方差 $D(X)$ 的相合估计量**



### 56.区间估计

点估计，给出未知参数的估计值，不能反应估计精确程度

区间估计：估计出一个范围，并给出此范围包含参数 $\theta$ 真值的可信程度



例如天气：明天有80%的可能在27度-30度之间

一般用区间长度来刻画精确度，可信度不变的条件下，区间越短，精确度越高



**定义：**

设总体 $\Lambda$ 的分布函数为 $F(x; \theta; \ell)$，其中 $\theta$ 是未知的参数，$\theta \in \Theta$（$\Theta$ 是 $\theta$ 可能取值的范围）。



对于给定的置信度 $\alpha$（$0 < \alpha < 1$），由样本数据 $X_1, X_2, \dots, X_n$ 确定的两个统计量 $\hat{\theta} = \theta(X_1, X_2, \dots, X_n)$ 和 $\bar{\theta} = \theta(X_1, X_2, \dots, X_n)$，对于任意的 $\theta \in \Theta$，如果满足：
$$
P\left( \hat{\theta} < \theta < \bar{\theta} \right) \geq 1 - \alpha
$$
则称随机区间 $(\hat{\theta}, \bar{\theta})$ 是参数 $\theta$ 的**置信区间**，且置信水平为 $1 - \alpha$。

**置信区间的上下限和置信水平：**

•**$\hat{\theta}$**：表示置信水平为 $1 - \alpha$ 的双侧置信区间的置信下限。

•**$\bar{\theta}$**：表示置信水平为 $1 - \alpha$ 的双侧置信区间的置信上限。

•**$1 - \alpha$**：置信水平，通常取值为 95% 或 99%。置信水平 $1 - \alpha$ 反映了我们对于参数 $\theta$ 落在置信区间内的信心。



1. **置信区间的定义**：
   - 置信区间 $(\hat{\theta}, \bar{\theta})$ 是一个随机区间，$\theta$ 是待估计的总体参数，$\hat{\theta}$ 和 $\bar{\theta}$ 是根据样本数据计算的估计量（例如，样本均值）。置信区间反映了我们对总体参数的估计范围。
   
2. **置信度 $\alpha$ 和置信水平**：
   - **$\alpha$**: 置信水平的补充值，通常设定为 $\alpha = 0.05$，对应 95% 的置信水平。
   - **$1 - \alpha$**: 表示置信区间的置信水平。例如，$\alpha = 0.05$ 对应的置信水平是 95%。

3. **概率表达式**：
   - $P\left(\hat{\theta} < \theta < \bar{\theta} \right) \geq 1 - \alpha$ 说明在多个独立抽样的情况下，置信区间 $(\hat{\theta}, \bar{\theta})$ 包含真实总体参数 $\theta$ 的概率至少为 $1 - \alpha$。例如，95% 的置信区间意味着有 95% 的概率该区间包含真实参数。

4. **计算置信区间的方法**：
   - 对于**连续型数据**，可以根据样本数据计算出估计量 $\hat{\theta}$ 和 $\bar{\theta}$，并得到置信区间。
   - 对于**离散型数据**，可以通过样本数据的频率分布来构造置信区间，利用分布的特性进行推断。

5. **反复抽样和置信区间**：
   - 在反复抽样（例如，100次独立抽样）后，每个样本的估计值会落在某个区间内，称为**置信区间**。例如，若 $\alpha = 0.01$，即置信水平为 99%，那么100次抽样中有99次所构造的置信区间会包含真实值，只有1次不包含。

6. **实际例子**：
   - **若 $\alpha = 0.01$**，意味着每次抽样产生的置信区间有 99% 的概率包含真实参数值，而1% 的置信区间不包含真实值。
   - **若 $\alpha = 0.05$**，则置信区间有 95% 的概率包含真实参数值，5% 的概率不包含。



#### exp1：

设总体 $X \sim N(\mu, \sigma^2)$，其中 $\mu$ 未知，方差 $\sigma^2 > 0$ 已知，设 $X_1, X_2, \dots, X_n$ 是来自 $X$ 的一个样本，要求 $\mu$ 的置信水平为 $1 - \alpha$ 的置信区间。

**解：**

未知参数 $\mu=E[X]$

寻找一个统计量，与 $\mu$ 相关，分布确定，用以确定 $(\underline{\mu},\bar{\mu})$ 使得 $P\{\underline{\mu}<\mu<\bar{\mu}\}$

由点估计可知，$\bar{X}$ 是 $E[X]=\mu$ 的无偏估计，且是相合估计。



由 $X \sim N(\mu, \frac{\sigma^2}{n})$，可推得：
$$
\frac{X - \mu}{\sigma/\sqrt{n}} \sim N(0,1)
$$

因此，概率公式为：
$$
P\left\{ -z_{\alpha/2} < \frac{X - \mu}{\sigma/\sqrt{n}} < z_{\alpha/2} \right\} = 1 - \alpha
$$
此处，$z_{\alpha/2}$ 是标准正态分布的临界值。

$$
P\left\{ X - \frac{\sigma}{\sqrt{n}} z_{\alpha/2} < \mu < X + \frac{\sigma}{\sqrt{n}} z_{\alpha/2} \right\} = 1 - \alpha
$$
即 $\mu$ 的置信区间可以表示为：
$$
\left( \overline{X} - \frac{\sigma}{\sqrt{n}} z_{\alpha/2}, \overline{X} + \frac{\sigma}{\sqrt{n}} z_{\alpha/2} \right)
$$

例如：

设 $\alpha = 0.05$，即置信水平为 95%，$z_{\alpha/2} = 1.96$，样本均值为 $\overline{X}$，设已知总体标准差 $\sigma$，样本容量 $n = 16$。

则置信区间为：


$$
\left( \overline{X} - 1.96 \frac{\sigma}{4}, \overline{X} + 1.96 \frac{\sigma}{4} \right)
$$
这个区间的置信水平为 95%，表示95%的置信区间包含真实的总体均值 $\mu$。



**寻找一个样本 $X_1, X_2, \dots, X_n$ 和 $\theta$ 的函数**  
设 $W = W(X_1, X_2, \dots, X_n; \theta)$，$W$ 的分布不依赖于 $\theta$ 以及其他未知参数，称为具有这种性质的函数 $W$ 为枢轴量。例如：
$$
\frac{X - \mu}{\sigma/\sqrt{n}} \sim N(0,1)
$$
给定置信水平 $1 - \alpha$，确定两个常数 $a$ 和 $b$，使得$ P\left\{ a < W(X_1, X_2, \dots, X_n; \theta) < b \right\} = 1 - \alpha $  ，若能从 $a < W(X_1, X_2, \dots, X_n; \theta) < b$ 中解得与之等价的 $\theta_1 < \theta < \theta_2$，  其中 $\theta = \theta(X_1, X_2, \dots, X_n)$ 和 $\theta_2 = \theta(X_1, X_2, \dots, X_n)$ 都是统计量，那么 $(\theta_1, \theta_2)$ 就是 $\theta$ 的一个置信水平为 $1 - \alpha$ 的置信区间。



枢轴量 $W(X_1, X_2, \dots, X_n; \theta)$ 推导方法：  通过从 $\theta$ 的估计值着手推导。

示例：单正态总体 $X \sim N(\mu, \frac{\sigma^2}{n})$，样本 $X_1, X_2, \dots, X_n$

1. $X \sim N(\mu, \frac{\sigma^2}{n})$
2. $\frac{X - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$
3. $\frac{(n-1)s^2}{\sigma^2} \sim \chi^2(n-1)$
4. $\frac{X - \mu}{S/\sqrt{n}} \sim t(n-1)$



### 57.正态总体均值与方差的区间估计



