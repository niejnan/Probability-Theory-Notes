## Chap3.多维随机变量及其分布



### 25.二维随机变量的定义

**二维随机变量的定义**：

设随机试验 $$E$$ 的样本空间为 $$S = \{e\}$$。

设 $$X = X(e), Y = Y(e)$$ 是定义在 $$S$$ 上的随机变量，由它们构成的向量 $$(X, Y)$$ 称为二维随机变量。

若 $$X_1, X_2, \dots, X_n$$ 是定义在同一个样本空间 $$S$$ 上的 $$n$$ 个随机变量，则称 $$(X_1, X_2, \dots, X_n)$$ 是 $$n$$ 维随机变量，$$X_i$$ 称为第 $$i$$ 个分量。



**研究思路**：

二维随机变量 $$(X, Y)$$ 的性质不仅与 $$X$$ 及 $$Y$$ 有关，而且还依赖于它们二者的相互关系。



放在一起当做整体来研究 $(X,Y)$ 就是联合分布，包括联合分布律、联合概率密度函数、联合分布函数等



当做个体来研究，就是边缘分布，条件分布，考虑独立性



**二维随机变量的(联合)分布函数**

**定义**：设 $$(X, Y)$$ 是二维随机变量，对任意实数 $$x, y$$，称二维函数

$$
F(x, y) = P\left( (X \leq x) \cap (Y \leq y) \right) = P\{X \leq x, Y \leq y\}
$$

为二维随机变量 $$(X, Y)$$ 的（联合）分布函数。其中，$(x,y) R^2$



**注**：$$F(x, y)$$ 是事件 $$A = \{X \leq x\}$$ 和 $$B = \{Y \leq y\}$$ 同时发生的概率。



**概率意义：**

如果将 $$(X, Y)$$ 看成平面上的随机点的坐标，则分布函数 $$F(x, y)$$ 在 $$(x, y)$$ 处的函数值就是随机点 $$(X, Y)$$ 落在以点 $$(x, y)$$ 为顶点而位于该点下方的无穷矩形区域的概率。
$$
P\{X\leq x\,Y\leq y\}
$$
也就是 $X\leq x$， $Y\leq y$ 落在平面上的点。



**分布函数的性质**

定一议一，

1. 固定 $y$，当 $x_1\leq x_2$ 时，单调不减的函数

2. 有界性，极限的相关性质 $F(x,y)\leq 1$

固定 $$y$$ 时：

$$
\lim_{x \to -\infty} F(x, y) = 0
$$

固定 $$x$$ 时：

$$
\lim_{y \to -\infty} F(x, y) = 0
$$

$$
\lim_{x \to -\infty, y \to -\infty} F(x, y) = 0
$$

当 $$x \to +\infty$$ 和 $$y \to +\infty$$ 时：

$$
\lim_{x \to +\infty, y \to +\infty} F(x, y) = 1
$$

**结论**：

$$
\lim_{x \to +\infty} F(x, y) \quad \text{不确定}
$$

$$
\lim_{y \to +\infty} F(x, y) \quad \text{不确定}
$$

3. 关于 $x,y$ 右连续

4. 不等式性质

对于任意 $$x_1 < x_2, y_1 < y_2$$，

$$
\Delta = P\{x_1 < X \leq x_2, y_1 < Y \leq y_2\}
$$

$$
= P\{X \leq x_2, Y \leq y_2\} - P\{X \leq x_2, Y \leq y_1\}
$$

$$
- P\{X \leq x_1, Y \leq y_2\} + P\{X \leq x_1, Y \leq y_1\}
$$

$$
= F(x_2, y_2) - F(x_2, y_1) - F(x_1, y_2) + F(x_1, y_1)
$$

且 $$> 0$$。



### 26.二维离散型随机变量

**定义**：

若二维随机变量 $$(X, Y)$$ 只能取有限对值 $$(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$$，则称 $$(X, Y)$$ 为二维离散型随机变量。

**联合分布律**：

称 $$P\{X = x_i, Y = y_j\} = p_{ij}, \quad i, j = 1, 2, \dots$$ 为二维离散型随机变量 $$(X, Y)$$ 的（联合）分布律。

其表格形式为：

$$
\begin{array}{c|c c c \dots c}
Y & x_1 & x_2 & \dots & x_i & \dots \\
\hline
y_1 & p_{11} & p_{21} & \dots & p_{i1} & \dots \\
y_2 & p_{12} & p_{22} & \dots & p_{i2} & \dots \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
y_j & p_{1j} & p_{2j} & \dots & p_{ij} & \dots \\
\end{array}
$$
**性质：**

1. $$
   P_{ij}\geq0
   $$

2. $$
   \sum_{i=1}^{+\infty} \sum_{j=1}^{+\infty} p_{ij} = 1
   $$



设 $$(X, Y)$$ 的联合分布为 $$P\{X = x_i, Y = y_j\} = p_{ij}, \quad i, j = 1, 2, \dots$$，则 $$(X, Y)$$ 的联合分布函数为

$$
F(x, y) = P\{X \leq x, Y \leq y\} = \sum_{x_i \leq x} \sum_{y_j \leq y} P\{X = x_i, Y = y_j\}
$$

$$
= \sum_{x_i \leq x} \sum_{y_j \leq y} p_{ij}
$$

其中和式是对一切满足的 $$x_i \leq x, y_j \leq y$$ 的 $$i, j$$ 求和。

**注**：
1. $$F(x, y)$$ 是点 $$(x, y)$$ 的左下角四分之一平面上 $$X$$ 和 $$Y$$ 所有可能取值的概率的和。
2. $$ (X, Y) $$ 落入平面区域 $$G$$ 的概率等于 $$ (X, Y) $$ 在 $$G$$ 内所有可能值的概率和。这是计算概率、求随机变量函数分布的一个重要公式。



### 27.二维连续型随机变量

**定义**：对二维随机变量 $$(X, Y)$$ 的分布函数 $$F(x, y)$$

如果存在非负函数 $$f(x, y)$$，使得对任意 $$x, y$$

$$
F(x, y) = \int_{-\infty}^{y} \int_{-\infty}^{x} f(u, v) \, du \, dv, \quad -\infty < x < +\infty, -\infty < y < +\infty
$$

则称 $$(X, Y)$$ 是二维连续型随机变量，称 $$f(x, y)$$ 为 $$(X, Y)$$ 的联合概率密度，记为 $$(X, Y) \sim f(x, y)$$。



**$f(x,y)$ 的性质**

1. 非负
2. 整个坐标平面做二重积分=1



**联合分布函数、联合概率密度函数的性质**

设 $$(X, Y)$$ 的联合分布函数为 $$F(x, y)$$，概率密度为 $$f(x, y)$$，则：

1) $$F(x, y)$$ 是 $$(x, y)$$ 的二维连续函数；

$$
F(x, y) = \int_{-\infty}^{y} \int_{-\infty}^{x} f(u, v) \, du \, dv
$$

$$
\lim_{(x, y) \to (1, 2)} F(x, y) = F(1, 2).
$$

2) 在 $$f(x, y)$$ 的连续点处，有

$$
F(x, y) = \int_{-\infty}^{y} \int_{-\infty}^{x} f(u, v) \, du \, dv
$$

则 $$F(x, y)$$ 具有二阶偏导数，并且

$$
\frac{\partial^2 F(x, y)}{\partial x \partial y} = f(x, y).
$$

3) 若 $$F(x, y)$$ 可导，则 $$(X, Y)$$ 是二维连续型随机变量，且 $$ \frac{\partial^2 F(x, y)}{\partial x \partial y} $$ 是它的一个概率密度。

4) 设 $$G$$ 是平面上的某个区域，则

$$
P\{(X, Y) \in G\} = \int \int_G f(x, y) \, dx \, dy.
$$



![截屏2025-02-19 23.52.58](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-19 23.52.58.png)

1. 

$$
f(x, y) \geq 0 \quad \Rightarrow \quad c > 0
$$

$$
1 = \int_0^{+\infty} \int_0^{+\infty} c \cdot e^{-(2x + y)} \, dx \, dy
$$

$$
= c \int_0^{+\infty} e^{-y} \, dy \int_0^{+\infty} e^{-2x} \, dx
$$

$$
= c \cdot \left[ \int_0^{\infty} e^{-y} \, dy \right] \cdot \left[ \int_0^{\infty} e^{-2x} \, dx \right]
$$

$$
= c \cdot \left[ \lim_{y \to \infty} (-e^{-y}) + 1 \right] \cdot \left[ \lim_{x \to \infty} (-\frac{1}{2} e^{-2x}) + \frac{1}{2} \right]
$$

$$
= \frac{1}{2}c \quad \Rightarrow \quad c = 2.
$$

2. 对于非零值：

$$
F(x, y) = \int_0^y \int_0^x f(u, v) \, du \, dv
$$

$$
= \int_0^y \int_0^x e^{-(2u + v)} \, du \, dv
$$

$$
= 2 \int_0^y e^{-v} \int_0^x e^{-2u} \, du \, dv
$$

$$
= (1 - e^{-y}) \cdot (1 - e^{-2x})
$$

$$
F(x, y) = 
\begin{cases}
(1 - e^{-y})(1 - e^{-2x}), & x > 0, y > 0 \\
0, & \text{其他}
\end{cases}
$$

3. $$ D: 0 < x < +\infty, 0 < y < x $$

$$
P\{Y \leq X\} = \int_0^\infty \int_0^x f(x, y) \, dx \, dy
$$

$$
 = 2 \int_0^\infty e^{-2x} \, dx \int_0^x e^{-y} \, dy 
$$

$$
= \frac{1}{3} 
$$

### 28.边缘分布

已知二维随机变量 $$(X, Y)$$ 的联合分布函数为 $$F(x, y)$$，而 $$X, Y$$ 都是二维随机变量，各自也有分布函数，将其分别记为 $$F_X(x)$$，$$F_Y(y)$$，分布称为二维随机变量 $$(X, Y)$$ 关于 $$X$$ 和 $$Y$$ 的边缘分布函数。



**$(X,Y)$ 的边缘概率密度**
$$
f_X=\int_{-\infin}^{+\infin}f(x,y) \ dy
$$

$$
f_Y=\int_{-\infin}^{+\infin}f(x,y) \ dx
$$



### 29.条件分布

#### 二维离散随机变量 $(X,Y)$ 的条件分布律

**定义**：已知二维离散型随机变量 $$(X, Y)$$ 的（联合）分布律为
$$
P\{X = x_i, Y = y_j\} = p_{ij}, \quad i, j = 1, 2, \dots
$$

$$(X, Y)$$ 关于 $$X$$ 和 $$Y$$ 的边缘分布律分别为：

$$
P\{X = x_i\} = \sum_{j=1}^{\infty} P\{X = x_i, Y = y_j\} = \sum_{j=1}^{\infty} p_{ij}, \quad i = 1, 2, \dots
$$

$$
P\{Y = y_j\} = \sum_{i=1}^{\infty} P\{X = x_i, Y = y_j\} = \sum_{i=1}^{\infty} p_{ij}, \quad j = 1, 2, \dots
$$

对于固定的 $$j$$，若 $$P\{Y = y_j\} > 0$$，则称为在 $$Y = y_j$$ 条件下随机变量 $$X$$ 的条件分布。

$$
P\{X = x_i \mid Y = y_j\} = \frac{P\{X = x_i, Y = y_j\}}{P\{Y = y_j\}} = \frac{p_{ij}}{p_j}, \quad i = 1, 2, \dots
$$

对于固定的 $$i$$，若 $$P\{X = x_i\} > 0$$，则称为在 $$X = x_i$$ 条件下随机变量 $$Y$$ 的条件分布。

$$
P\{Y = y_j \mid X = x_i\} = \frac{P\{X = x_i, Y = y_j\}}{P\{X = x_i\}} = \frac{p_{ij}}{p_i}, \quad j = 1, 2, \dots
$$


#### 二维连续随机变量 $(X,Y)$ 的条件分布

设二维连续型随机变量 $(X, Y)$ 的联合概率密度为 $f(x, y)$。

在 $Y = y$ 的条件下 $X$ 的条件概率密度为：
$$
 f_{X|Y}(x | y) = \frac{f(x, y)}{f_Y(y)} 
$$
在 $Y = y$ 的条件下，$X$ 的条件分布函数为：

$$
F_{X|Y}(x | y) = \int_{-\infty}^{x} f_{X|Y}(x | y) dx 
$$
在 $X = x$ 的条件下，$Y$ 的条件概率密度为：

$$
f_{Y|X}(y | x) = \frac{f(x, y)}{f_X(x)}
$$
在 $X = x$ 的条件下，$Y$ 的条件分布函数为：

$$
 F_{Y|X}(y | x) = \int_{-\infty}^{y} f_{Y|X}(y | x) dy 
$$


联合分布 = 边缘分布 × 条件分布



### 30.相互独立的随机变量

#### 二维离散型随机变量 $(X,Y)$ 的相互独立性

1. 称 $X$ 和 $Y$ 相互独立：如果对于 $(X, Y)$ 所有可能取的值 $(x_i, y_j)$ 有：

$$
 P\{X = x_i, Y = y_j\} = P\{X = x_i\} P\{Y = y_j\} 
$$

注：
1) 联合分布 = 边缘分布乘以条件分布，$i = 1, 2, \dots, n$，$j = 1, 2, \dots, m$
2) $n \times m$ 个事件同时发生。

2. $X$ 和 $Y$ 相互独立等价于：对于所有的 $x, y$，

$$
P\{X \leq x, Y \leq y\} = P\{X \leq x\} P\{Y \leq y\}
$$

$$
F(x, y) = F_X(x) F_Y(y)
$$



#### 二维连续型随机变量 $(X,Y)$ 的相互独立性

1. 称 $X$ 和 $Y$ 相互独立：如果对于所有的 $x, y$，

$$
 f(x, y) = f_X(x) f_Y(y) \quad \forall x, y 
$$

注：
1) 上式对于所有的 $x, y$ 都成立。
2) 且有 $f(x, y)$，即 $f_X(x), f_Y(y)$ 存在。



2. $X$ 和 $Y$ 相互独立等价于：对于所有的 $x, y$，

$$
 P\{X \leq x, Y \leq y\} = P\{X \leq x\} P\{Y \leq y\} 
$$

$$
 F(x, y) = F_X(x) F_Y(y) 
$$



### 31.二维正态分布与二维均匀分布

#### 二维正态分布

定义：设二维随机变量 $(X, Y)$ 的概率密度为：

$$
f(x, y) = \frac{1}{2 \pi \sigma_1 \sigma_2 \sqrt{1 - \rho^2}} \exp \left\{ - \frac{1}{2(1 - \rho^2)} \left[ \frac{(x - \mu_1)^2}{\sigma_1^2} - 2 \rho \frac{(x - \mu_1)(y - \mu_2)}{\sigma_1 \sigma_2} + \frac{(y - \mu_2)^2}{\sigma_2^2} \right] \right\}
$$

其中，$\mu_1, \mu_2, \sigma_1, \sigma_2, \rho$ 都是常数，且 $\sigma_1 > 0, \sigma_2 > 0$，$-1 < \rho < 1$，则称 $(X, Y)$ 的服从参数为 $\mu_1, \mu_2, \sigma_1, \sigma_2, \rho$ 的二维正态分布，记为：

$$
(X, Y) \sim N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)
$$
当 $\rho=0$，表明 $X,Y$ 相互独立



#### 二维均匀分布

设 $G$ 是平面上的有界区域，其面积为 $A$。若二维随机变量 $(X, Y)$ 具有概率密度：

$$
f(x, y) = \begin{cases}
\frac{1}{A}, & (x, y) \in G \\
0, & \text{其他}
\end{cases}
$$

则称 $(X, Y)$ 在 $G$ 上服从均匀分布。

例：$(X, Y)$ 在 $G: x^2 + y^2 \leq 1$ 上服从的分布：

$$
f(x, y) = \begin{cases}
\frac{1}{\pi}, & x^2 + y^2 \leq 1 \\
0, & \text{其他}
\end{cases}
$$

$x^2 + y^2 = 1$ 表示单位圆。



### 32.两个随机变量的函数的分布(离散型)

太简单了，没什么好说的



### 33.两个随机变量和的分布(连续型)

**和的分布：**

设 $(X, Y)$ 的概率密度为 $f(x, y)$，则 $Z = X + Y$ 仍为连续型随机变量，其概率密度为：
$$
f_Z(z) = \int_{-\infty}^{+\infty} f(x, z - x) dx
$$

$$
f_Z(z) = \int_{-\infty}^{+\infty} f(z - y, y) dy 
$$

注：若 $X, Y$ 独立，则：

$$
f_Z(z) = \int_{-\infty}^{+\infty} f_X(x) f_Y(z - x)dx
$$

$$
f_Z(z) = \int_{-\infty}^{+\infty} f_X(z-y) f_Y(y)dy
$$



**证明：**

设 $Z = X + Y$ 的分布函数 $F_Z(z)$
$$
F_Z(z) = P\{ Z \leq z \} = P\{ X + Y \leq z \} = P\{ (X, Y) \in G \}
$$

其中，$G: X + Y \leq z$

$$
F_Z(z) = \int_{-\infty}^{+\infty} \left[ \int_{-\infty}^{z - y} f(x, y) dx \right] dy
$$

转换得到：

$$
F_Z(z) = \int_{-\infty}^{z} \left[ \int_{-\infty}^{\infty} f(u - y, y) du \right] dy
$$

因此，$Z = X + Y$ 的分布函数为：

$$
F_Z(z) = \int_{-\infty}^{z} \left[ \int_{-\infty}^{\infty} f(u - y, y) dy \right] du
$$

从而得到 $Z = X + Y$ 的概率密度为：

$$
f_Z(z) = \int_{-\infty}^{\infty} f(z - y, y) dy
$$

同样，$f_Z(z)$ 也可以写为：

$$
f_Z(z) = \int_{-\infty}^{\infty} f(x, z - x) dx
$$


**注：**
1. 若 $X, Y$ 相互独立，且 $X \sim N(\mu_1, \sigma_1^2)$，$Y \sim N(\mu_2, \sigma_2^2)$，则 $X + Y \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$。

2. 若 $n$ 个独立随机变量的和服从正态分布：

    线性组合也服从正态分布。

若 $X_1, X_2, \dots, X_n$ 相互独立，且 $X_i \sim N(\mu_i, \sigma_i^2)$，$i = 1, 2, \dots, n$，则：

$$
X_1 + X_2 + \cdots + X_n \sim N(M_1 + M_2 + \cdots + M_n, \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_n^2)
$$

令：

$$
S_4 \triangleq a_1 X_1 + a_2 X_2 + \cdots + a_n X_n \sim N(\ldots, \ldots)
$$


#### 34.两个随机变量商和积的分布(连续型)

**商的分布：**

设 $(X, Y)$ 是二维连续型随机变量，它具有概率密度为 $f(x, y)$，则 $Z = \frac{Y}{X}$ 仍为连续型随机变量，其概率密度为：
$$
f_{Y/X}(z) = \int_{-\infty}^{+\infty} |x| f(x, xz) dx
$$

若 $X, Y$ 独立，则：

$$
f_{Y/X}(z) = \int_{-\infty}^{+\infty} |x| f_X(x) f_Y(xz) dx
$$


**积的分布：**

设二维连续型随机变量 $(X, Y)$ 的概率密度为 $f(x, y)$，则 $Z = X \cdot Y$ 仍为连续型随机变量，其概率密度为：
$$
f_{XY}(z) = \int_{-\infty}^{+\infty} \frac{1}{|x|} f(x, \frac{z}{x}) dx
$$

若 $X, Y$ 独立，则：

$$
f_{XY}(z) = \int_{-\infty}^{+\infty} \frac{1}{|x|} f_X(x) f_Y\left( \frac{z}{x} \right) dx
$$


### 35.两个随机变量的最大值、最小值的分布(连续型)

#### 最大值分布：

设 $X, Y$ 的分布函数分别为 $F_X(x), F_Y(y)$，求 $Z = \max\{X, Y\}$​ 的分布函数及概率密度函数。
$$
F_Z(z) = P\{Z \leq z\} = P\{\max\{X, Y\} \leq z\}
$$

$$
= P\{X \leq z, Y \leq z\}
$$

若 $X, Y$ 独立，则：

$$
P\{X \leq z, Y \leq z\} = F_X(z) \cdot F_Y(z)
$$

因此，$Z$ 的分布函数为：

$$
F_Z(z) = \left[ F(z) \right]^2
$$

**扩展：** 

设 $Z = \max\{X_1, X_2, \dots, X_n\}$
$$
F_Z(z) = P\{\max\{X_1, X_2, \dots, X_n\} \leq z\}
$$

$$
= P\{X_1 \leq z, X_2 \leq z, \dots, X_n \leq z\}
$$

若 $X_1, X_2, \dots, X_n$ 独立，则：

$$
P\{X_1 \leq z\} \cdot P\{X_2 \leq z\} \dots P\{X_n \leq z\}
$$

因此，$Z$ 的分布函数为：

$$
F_Z(z) = \left[ F(z) \right]^n
$$



**求 $Z = \max\{X, Y\}$ 的 $f_Z(z)$：**

若 $X, Y$ 相互独立，则 $F_Z(z) = \left[ F(z) \right]^2$，此时：

$$
f_Z(z) = \left[ F_Z(z) \right]'_z = \left[ F(z)^2 \right]'_z
$$

$$
= 2 F(z) \cdot F'(z)
$$

$$
= 2 F(z) \cdot f(z)
$$





#### 最小值分布：

设 $X, Y$ 的分布函数分别为 $F_X(x), F_Y(y)$，求 $Z = \min\{X, Y\}$ 的分布函数及概率密度函数。
$$
F_Z(z) = P\{Z \leq z\} = P\{\min\{X, Y\} \leq z\}
$$

$$
= 1 - P\{\min\{X, Y\} > z\}
$$

$$
= 1 - P\{X > z, Y > z\}
$$

若 $X, Y$ 独立，则：

$$
= 1 - P\{X > z\} \cdot P\{Y > z\}
$$

$$
= 1 - [1 - F_X(z)] \cdot [1 - F_Y(z)]
$$

因此，$Z$ 的分布函数为：在 $X,Y$ 独立且同分布的情况下

$$
F_Z(z) = 1 - [1 - F(z)]^2 \quad (F(z): X(Y) 的分布函数)
$$



**扩展：**

 设 $Z = \min\{X_1, X_2, \dots, X_n\}$，$X_1, X_2, \dots, X_n$ 独立同分布，则：
$$
F_Z(z) = 1 - [1 - F(z)]^n
$$

$$
f_Z(z) = \left[ F_Z(z) \right]'_z = \left[ 1 - [1 - F(z)]^n \right]'_z
$$

$$
= -n[1 - F(z)]^{n-1} \cdot [-f(z)]
$$

$$
= n f(z) \cdot [1 - F(z)]^{n-1}
$$











