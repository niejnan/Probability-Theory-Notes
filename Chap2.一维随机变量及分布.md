## Chap2.一维随机变量及分布



### 13.随机变量

**随机变量的定义**

设随机试验 $E$ 的样本空间为 $S = \{e\}$.

$X = X(e)$ 是定义在样本空间 $S$ 上的实值单值函数，称 $X = X(e)$ 为随机变量。

约定：$X, Y, Z, W \to$ 随机变量，$x, y, z, w \to$ 变量。



随机变量示意图：

![截屏2025-02-19 11.31.44](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 11.31.44.png)

随机变量与普通函数的区别：

1. 随机变量：定义域：$S={1,2,3,4,5,6}$ 是样本空间，值域：实数
2. 普通函数：定义域：数集，值域：实数

随机变量的取值是随机的，普通函数取值是确定的

随机变量$X$ 的本质是对事件的描述，例如 $X=x$，$X\leq x$



用随机变量 研究 随机现象，随机线性 翻译称随机变量



### 14.离散型随机变量及其分布律

离散--有限个

**定义：** 

若随机变量 $X$ 全部可能取到的值是有限个，或者可列无穷多个，则称 $X$ 为离散型随机变量

**分布律**：

设离散型随机变量 $X$ 所有可能的值为 $x_k (k=1, 2, \dots)$，$X$ 取各个可能值的概率，即事件 $\{X = x_k\}$ 的概率为 $P\{X = x_k\} = p_k, k = 1, 2, \dots$，称其为离散型随机变量 $X$ 的分布律。



**分布律的性质**：

1. $p_k \geq 0, \quad (k = 1, 2, \dots)$;
2. $\sum_{k=1}^{\infty} p_k = 1$。



**分布律表格形式**：

| $X$   | $x_1$ | $x_2$ | $\dots$ | $x_n$ | $\dots$ |
| ----- | ----- | ----- | ------- | ----- | ------- |
| $p_k$ | $p_1$ | $p_2$ | $\dots$ | $p_n$ | $\dots$ |



### 15.0-1分布

**定义：** 设随机变量 $X$ 只可能取到 0 和 1 两个值，它的分布律是：
$$
P\{X = k\} = p^k(1 - p)^{1 - k}, \quad k = 0, 1 \quad (0 < p < 1)
$$
则称 $X$ 服从参数为 $p$ 的 (0-1 分布) 或两点分布。



### 16.二项分布(伯努利)

**1. 伯努利试验**：

设试验 $E$ 只有两个可能结果：$A$ 及 $\overline{A}$，则称 $E$ 为伯努利试验。

设 $P(A) = p \quad (0 < p < 1)$。

将 $E$ 独立重复进行 $n$ 次，则称这一串重复的独立试验为 $n$ 次伯努利试验。

**注**：

**“重复”** 是指在每次试验中 $P(A) = p$ 保持不变；

**“独立”** 是指在各次试验的结果互不影响。



**2. 二项分布**：

以 $X$ 表示重伯努利试验中事件 $A$ 发生的次数，$P(A) = p \quad (0 < p < 1)$，求 $X$ 分布律。



**定义**：若 $X$ 的分布律为
$$
P\{X = k\} = C_n^k p^k (1 - p)^{n - k}, \quad k = 0, 1, 2, \dots, n
$$

则称 $X$ 服从参数为 $n$ 和 $p$ 的二项分布，记为
$$
X \sim b(n, p)
$$




0-1 分布：$P\{X=k\}=p^k(1-p)^{1-k}$

二项分布：$P\{X=k\}=C_n^kp^k(1-p)^{n-k}$

当 $n=1$ 时，二项分布就是0-1分布

可以把二项分布看作是：0-1分布的推广，也可以理解为0-1分布是二项分布的特殊情况。
$$
(a+b)^n=\sum_{k=0}^{n}C_n^ka^kb^{n-k}
$$


![截屏2025-02-19 12.19.45](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 12.19.45.png)

$X\sim B(400,0.02)$
$$
P\{X\geq2\}=1-P\{X<2\}
$$

$$
=1-P\{X=1\}-P\{X=2\}
$$



### 17.泊松分布

泊松分布可以看作是二项分布当 $n\rightarrow \infin$ 的情况

**泊松分布定义**：若随机变量 $X$ 的分布律为
$$
P\{X = k\} = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots
$$
其中 $\lambda > 0$ 是常数，则称随机变量 $X$ 服从参数为 $\lambda$ 的泊松分布，记为
$$
X \sim P(\lambda).
$$

并且有
$$
\sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x.
$$


**泊松定理**

设 $\lambda > 0$ 是一个常数，$n$ 是任意正整数，设 $np_n = \lambda$，则对于任一固定的非负整数 $k$，有

$$
\lim_{n \to \infty} C_n^k p_n^k (1 - p_n)^{n - k} = \frac{\lambda^k e^{-\lambda}}{k!}
$$

即
$$
P\{X = k\} \to P\{X = k\} \quad \text{当} \quad n \to \infty \quad \text{且} \quad \lambda = n p_n.
$$
泊松定理告诉我们，当 $n\rightarrow \infin$ 时，二项分布近似于泊松分布 $\lambda=np$

当 $n$ 很大的时候， $p$ 很小的时候，可以用泊松分布近似二项分布
$$
P=\{X=k\}=C_n^kp^k(1-p)^{n-k}=\frac{\lambda^ke^{-\lambda}}{k!},\lambda=np
$$


![截屏2025-02-19 12.46.19](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 12.46.19.png)

1. 二项分布

$$
P\{X \geq 2\} = 1 - P\{X = 0\} - P\{X = 1\}
$$
$$
= 1 - 0.999^{1000} - (1000 \cdot 0.01^{1} - 0.999^{999})
$$
$$
= 1 - 0.999^{1000} - 0.999^{999} \approx 0.2642411.
$$

2. 泊松定理：$n = 1000, p = 0.001, \lambda = np = 1$.

$$
P\{X \geq 2\} = 1 - P\{X = 0\} - P\{X = 1\}
$$
$$
\approx 1 - e^{-1} - e^{-1} = 1 - 2 e^{-1} \approx 0.2642411.
$$

注：一般来说，当$X \sim b(n, p)$，$n \geq 20, p \leq 0.05$，可以用泊松定理近似二项分布



### 18.几何分布、超几何分布

#### 几何分布

**定义**：

在独立重复试验中，试验次数预先不能确定。设每次试验成功的概率为 $p$，将实验进行到成功为止，以 $X$ 表示所需的试验次数，则 $X$ 的分布律为
$$
P\{X = k\} = (1 - p)^{k - 1} p, \quad k = 1, 2, \dots
$$

则称随机变量 $X$ 服从参数为 $p$ 的几何分布。



#### 超几何分布

**定义**：

从 $N$ 件产品（其中含次品 $M$ 件）中任取 $n$ 件，以 $X$ 表示取得的次品数，则 $X$ 的分布律为
$$
P\{X = k\} = \frac{C_M^k C_{N-M}^{n-k}}{C_N^n}, \quad 0 \leq k \leq n \leq N, \quad k \leq M
$$

则称随机变量 $X$ 服从参数为 $(N, M, n)$ 的超几何分布。





### 19.随机变量的分布函数

对于离散型随机变量，可以一一列举，用分布律描述

实际问题：$X$ 是测量长度的误差：$P\{x_1<X \leq x_2\}=P\{X\leq x_2\}-P\{X\leq x_1\}$

右端点的函数值，减去左端点的函数值



**定义**：设 $X$ 是一个随机变量，$x$ 是任意实数，函数

$$
F(x) = P\{X \leq x\}, \quad -\infty < x < +\infty
$$

称为 $X$ 的分布函数。

1. **定义域**：$x \in (-\infty, +\infty)$。
2. **值域**：$0 \leq F(x) \leq 1$。
3. **性质**：若 $x_1, x_2 \quad (x_1 < x_2)$，则
$$
P\{x_1 < X \leq x_2\} = P\{X \leq x_2\} - P\{X \leq x_1\}
$$
4. **分布函数：** 高等数学



**几何表示**：

将 $X$ 看成数轴上的随机点的坐标，那么分布函数 $F(x)$ 在 $x$ 处的函数值就表示 $X$ 落在区间 $(-\infty, x]$​ 上的概率。

![截屏2025-02-19 13.21.32](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 13.21.32.png)

1. $F(x)$ 是单调不减函数
2. $0\leq F(x)\leq 1$；$\lim_{x\rightarrow+{\infin}}F(x)=1$；$\lim_{x\rightarrow-{\infin}}F(x)=0$

3. $F(x)$ 右连续



![截屏2025-02-19 13.28.12](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 13.28.12.png)

$F(x)$ 是左闭右开

$$
P\left\{ \frac{3}{2} < X \leq \frac{5}{2} \right\} = F\left(\frac{5}{2}\right) - F\left(\frac{3}{2}\right) = \frac{3}{4} - \frac{1}{4} = \frac{2}{4}
$$

$$
P\{ 2 \leq X \leq 3 \} = P\{ X = 2 \} + P\{ 2 < X < 3 \}
$$
$$
= \frac{1}{2} + F(3) - F(2) = \frac{3}{4}.
$$



**离散型随机变量分布函数的性质**

离散型 $X$：$P\{X = x_k\} = p_k, \quad k = 1, 2, 3, \dots$，则

$$
F(x) = P\{X \leq x\} = \sum_{x_k \leq x} P\{X = x_k\}
$$

1. **$F(x)$ 图形**：一条阶梯曲线。
2. **$F(x)$ 在 $x = x_k$ 处有跳跃值**：$P_k = P\{X = x_k\}$。
3. **$F(x)$：分段函数**；分段区间，左闭右开，$n$ 段函数，$n+1$ 段区间



### 20.连续型随机变量及其概率密度

$F(x)=\int_{-\infin}^{x}f(t)dt$



**定义**：

若对于随机变量 $X$ 的分布函数 $F(x)$，存在非负可积函数 $f(x)$，使得对于任意实数 $x$，有
$$
F(x) = \int_{-\infty}^{x} f(t) \, dt,
$$

则称 $X$ 为连续型随机变量，其中函数 $f(x)$ 称为 $X$ 的概率密度函数，简称为概率密度。



**概率密度函数 $f(x)$ 性质：**

1. $f(x)\geq0$

2. $\int_{-\infin}^{+\infin}f(x)dx= 1$，几何意义是介于曲线 $y=f(x)$ 与 $x$ 轴之间的区域面积是1

3. 
   $$
   P\{x_1 < X \leq x_2\} = P\{X \leq x_2\} - P\{X \leq x_1\}
   $$
   $$
   = F(x_2) - F(x_1) = \int_{x_1}^{x_2} f(x) \, dx
   $$

   **几何意义**：

   $X$ 落在区间 $(x_1, x_2]$ 的概率等于区间 $(x_1, x_2]$ 上曲线 $y = f(x)$​ 之下的曲边梯形的面积。

4. 若 $f(x)$ 在 $x_0$ 点处连续，则 $F'(x_0)=f(x_0)$



**连续型随机变量的四个特性：**

1. 连续型随机变量的分布函数 $F(x)$ 一定连续；

$$
F(x) = \int_{-\infty}^{x} f(t) \, dt \quad \text{连续}.
$$

2. 若 $f(x)$ 在点 $x_0$ 处连续，则必有

$$
F'(x_0) = f(x_0)
$$

3. 设 $X$ 为连续型随机变量，$a$ 为常数， 则

$$
P\{X = a\} = 0;
$$

**证明**：设连续 $X$ 的分布函数为 $F(x)$，全 $\Delta x > 0$
$$
\{X = a\} \subset \{a - \Delta x < X \leq a\}
$$

故

$$
0 \leq P\{X = a\} \leq P\{a - \Delta x < X \leq a\} = F(a) - F(a - \Delta x)
$$

随着 $\Delta x \to 0$，得

$$
P\{X = a\} = 0.
$$

> 这个的意义在于，对于连续型随机变量，在确定的一点处的概率为0

4. 设 $X$ 为连续型随机变量，则

$$
P\{a < X < b\} = P\{a \leq X \leq b\} = P\{a < X \leq b\}
$$

$$
= \int_{a}^{b} f(x) \, dx = F(b) - F(a)
$$

> 由3就可以知道，对于连续型随机变量，端点处的概率都是0，所以不需要在意等号取不取

 

### 21.均匀分布

**定义**：若随机变量 $X$ 的概率密度为

$$
f(x) =
\begin{cases}
\frac{1}{b - a}, & \text{if } a < x < b \\
0, & \text{otherwise}
\end{cases}
$$

则称 $X$ 在区间 $(a, b)$ 上服从均匀分布，记为 $X \sim U(a, b)$，其分布函数为

$$
F(x) =
\begin{cases}
0, & x < a \\
\frac{x - a}{b - a}, & a \leq x \leq b \\
1, & x \geq b
\end{cases}
$$

例如，$X \sim U(2, 5)$，则

$$
f(x) =
\begin{cases}
\frac{1}{3}, & 2 \leq x \leq 5 \\
0, & \text{otherwise}
\end{cases}
$$

$$
F(x) =
\begin{cases}
0, & x < 2 \\
\frac{x - 2}{3}, & 2 \leq x \leq 5 \\
1, & x > 5
\end{cases}
$$

$X$ 在区间 $(a, b)$ 上服从均匀分布具有下述意义的等可能性：

它落在区间 $(a, b)$ 中任意长度的子区间内的可能性相同；或者它落在 $(a, b)$ 的子区间内的概率只依赖于子区间的长度而与子区间的位置无关。



### 22.正态分布

**定义**：若随机变量 $X$ 的概率密度为

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}, \quad -\infty < x < +\infty
$$

其中 $\mu, \sigma$ 为常数，且 $\sigma > 0$，则称 $X$ 服从参数为 $\mu, \sigma$ 的正态分布，记为 $X \sim N(\mu, \sigma^2)$。

其分布函数为
$$
F(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(t - \mu)^2}{2\sigma^2}} \, dt, \quad -\infty < x < +\infty.
$$

不能积分出来是因为 $e^{-x^2}$ 的积分不能用初等函数表示



$\mu,\sigma$ 变化情形：

1. 固定 $\sigma$ ，变化 $\mu$ 的取值，图像沿着 $x$ 轴平移，形状不变
2. 固定 $\mu$，变化 $\sigma$ 取值：$f_{max}(\mu)=\frac{1}{\sqrt{2\pi} \sigma}$

也就是说，$\sigma$ 越大， $f_{max}$ 越小，对应图像就越平缓



![截屏2025-02-19 21.51.41](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 21.51.41.png)



**标准正态分布**
$$
X \sim N(0, 1), \quad \varphi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}, \quad -\infty < x < +\infty \quad \Leftrightarrow \quad x = 0
$$

$$
\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} \, dt \quad
$$

$$
\Phi(-x) = 1 - \Phi(x) \quad \Phi(0) = \frac{1}{2}= P\{X \leq 0\}
$$



**若 $x \sim N(\mu,\sigma^2)$ 怎么求 $X$ 相关事件的概率** 

1. 数形结合
2. 将 $N(\mu,\sigma^2)$ 转化为 $N(0,1)$​



**引理**：若 $X \sim N(\mu,\sigma^2)$，则 $\frac{X-\mu}{\sigma}\sim N(0,1)$



**重点总结**：设 $X \sim N(\mu, \sigma^2)$，则：

1) 
$$
F(x) = P\{X \leq x\} = P\left\{ \frac{X - \mu}{\sigma} \leq \frac{x - \mu}{\sigma} \right\} = \Phi\left( \frac{x - \mu}{\sigma} \right) \quad \text{（标准化）}
$$

2) 
$$
P\{x_1 < X \leq x_2\} = F(x_2) - F(x_1) = \Phi\left( \frac{x_2 - \mu}{\sigma} \right) - \Phi\left( \frac{x_1 - \mu}{\sigma} \right)
$$

例如，$X \sim N(1, 4)$，求 $P\{0 < X \leq 1.6\}$：

$$
P\{0 < X \leq 1.6\} = \Phi\left( \frac{1.6 - 1}{2} \right) - \Phi\left( \frac{0 - 1}{2} \right)
$$
$$
= \Phi(0.3) - \Phi(-0.5) = \Phi(0.3) + \Phi(0.5) - 1
$$



![截屏2025-02-19 22.04.49](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.04.49.png)

1. $P\{X<89\}=\Phi(\frac{89-90}{0.5})=\Phi(-2)=1-\Phi(2)$

2. $P(X<80)=\Phi(\frac{80-d}{0.5})<0.01$，即

$$
\Phi(\frac{80-d}{0.5})<0.01
$$



### 23.指数分布

**定义**：若随机变量 $X$  的概率密度为
$$
f(x) =
\begin{cases}
\frac{1}{\theta} e^{-x/\theta}, & x > 0 \\
0, & \text{其他}
\end{cases}
$$

其中 $\theta > 0$ 为常数，则称 $X$  服从参数为 $\theta$  的指数分布。其分布函数为：

$$
F(x) =
\begin{cases}
0, & x < 0 \\
1 - e^{-x/\theta}, & x \geq 0
\end{cases}
$$


**指数分布无记忆性：**

对于任意 $s, t > 0$，有

$$
P\{X > s + t \mid X > s\} = P\{X > t\}.
$$
**证明**：  

$$
I = \frac{P\{X > s + t \} \cdot P\{X > s\}}{P\{X > s\}} 
$$

$$
= \frac{P\{X > s + t\}}{P\{X > s\}} = \frac{1 - P\{X \leq s + t\}}{1 - P\{X \leq s\}} = \frac{1 - F(s + t)}{1 - F(s)}
$$

$$
= \frac{e^{-\frac{s+t}{\theta}}}{e^{-\frac{s}{\theta}}} = e^{-\frac{t}{\theta}} = P\{X > t\}
$$



 ![截屏2025-02-19 22.27.14](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.27.14.png)



$P\{X\leq 10\}=\int_{0}^{10}f_X(x)dx=1-e^{-2}$，$P\{X>10\}=e^{-2}$

$Y$ 是服从 $B(5,e^{-2})$ 的二项分布，写出分布律即可



### 24.随机变量的函数的分布

比如实际中，关心的随机变量不能是由测量得到的，他是由某个能直接测量的随机变量的函数。

比如：求圆锥直径 $d$，关心截面面积，通过 $d$ 分布求 $S$ 的分布



**定义**：设 $$X$$ 是随机变量，函数 $$y = g(x)$$，则以随机变量 $$X$$ 作为自变量的函数 $$Y = g(X)$$ 也是随机变量，称之为随机变量 $$X$$ 的函数。

例如：

$$
Y = aX + c
$$

$$
Y = |X - a|
$$

$$
Y =
\begin{cases}
X, & X \leq 1 \\
1, & X > 1
\end{cases}
$$

问题：已知 $X$ 的概率分布，求 $Y=g(X)$ 的概率分布。



#### 离散型计算方法![截屏2025-02-19 22.42.48](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.42.48.png)

![截屏2025-02-19 22.43.04](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.43.04.png)



#### 连续型计算方法

**分布函数求导法**：

设 $$X = f_X(x), F_X(x)$$，而 $$Y = g(X)$$，则 $$Y$$ 的概率分布为 $$f_Y(y)$$ 和 $$F_Y(y)$$。

1) 由分布函数定义，设 $$Y = g(X)$$ 为随机变量 $$X$$ 的函数：

$$
F_Y(y) = P\{Y \leq y\} = P\{g(X) \leq y\} = \int_{g^{-1}(y)}^{\infty} f_X(x) dx
$$

（写出分布方式，不必求解）

2) $$f_Y(y) = F_Y'(y)$$。



![截屏2025-02-19 22.47.31](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.47.31.png)

**解**：分布函数求导法

1) 
$$
F_Y(y) = P\{Y \leq y\} = P\{2X + 8 \leq y\} = P\{X \leq \frac{y - 8}{2}\}
$$

$$
= \int_{-\infty}^{\frac{y - 8}{2}} f_X(x) dx = F_X\left(\frac{y - 8}{2}\right)
$$

2) 
$$
f_Y(y) = F_Y'(y) = \left[F_X\left(\frac{y - 8}{2}\right)\right]'_y
$$

$$
= f_X\left(\frac{y - 8}{2}\right) \cdot \frac{1}{2}
$$

$$
= \begin{cases}
\frac{1}{2} \cdot \frac{1}{8} \cdot (y - 8), & 0 < \frac{y - 8}{2} < 4 \\
0, & \text{其他}
\end{cases}
$$

$$
= \begin{cases}
\frac{y - 8}{32}, & 8 < y < 16 \\
0, & \text{其他}
\end{cases}
$$



![截屏2025-02-19 22.50.39](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 22.50.39.png)

1) 
$$
F_Y(y) = P\{Y \leq y\} = P\{X^2 \leq y\}, \quad y \in (-\infty, +\infty)
$$

当 $y < 0$  时，$$ X^2 \leq y $$ 不可能发生，因此

$$
F_Y(y) = 0
$$

当 $y > 0$ 时，

$$
F_Y(y) = P\{X^2 \leq y\} = P\{-\sqrt{y} \leq X \leq \sqrt{y}\}
$$

$$
= F_X(\sqrt{y}) - F_X(-\sqrt{y}).
$$

总结：
$$
F_Y(y) =
\begin{cases}
0, & y < 0 \\
F_X(\sqrt{y}) - F_X(-\sqrt{y}), & y \geq 0
\end{cases}
$$

2) 
$$
f_Y(y) = F_Y'(y) =
\begin{cases}
0, & y < 0 \\
f_X(\sqrt{y}) \cdot (\sqrt{y})' - f_X(-\sqrt{y}) \cdot (-\sqrt{y})', & y \geq 0
\end{cases}
$$

$$
= \begin{cases}
0, & y < 0 \\
\frac{1}{2\sqrt{y}} [f_X(\sqrt{y}) + f_X(-\sqrt{y})], & y \geq 0
\end{cases}
$$



连续型随机变量的函数不一定是连续型随机变量，

但是若是单调可导的函数，则连续型随机变量的函数是连续型随机变量



若 $X \sim N(\mu,\sigma^2)$，若 $Y=aX+b$，则 $Y \sim N(a\mu+b,(a \sigma)^2)$

