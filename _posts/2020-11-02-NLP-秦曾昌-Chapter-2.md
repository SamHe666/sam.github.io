---
layout: post
title: "NLP-秦曾昌 Chapter 2"
date:   2020-11-02
tags: [NLP]
toc: true
author: Sam
---



## Mathematical Foundation

### 1. Functions

+ 指数函数 $y=e^{x}$
+ 对数函数 $y=log_{e}(x)$, 它和指数函数互为反函数
+ Sigmoid函数 $S(x) = \frac{1}{1+e^{-x}} = \frac{e^{x}}{e^x+1}$,这个函数有特别好的数学性质，把整个实数域映射到0到1之间，经常被用作激活函数 
+ Rectifier $f(x) = x^+ = max(0, x)$

<img src="https://i.loli.net/2020/11/02/tsDJKXMgVCB5uwY.png" alt="1" style="zoom: 50%;" />



### 2. Linear Algebra

#### 2.1. Norm of Vectors 向量的范式

向量范式用来表示向量的长度，它可以通过向量的内积来定义，在不同定义的向量内积下，向量的长度也会不同，不管其如何定义的，总满足以下几个性质：
+ $\lVert x \rVert \geq 0$, with equality if and only if x = 0 (0向量)
+ $\lVert \alpha x \rVert= \left| \alpha \right| \lVert x \rVert$
+ $\lVert x + y \rVert \leq \lVert x \rVert + \lVert y \rVert$ (the traiangle inequality)



以下是两个常用的向量范式
+ $\ell_1$ is Manhattan Distance: $$ \lVert x \rVert_1 = \sum_{i=1}^{n} \left| x_i \right| $$

+ $\ell_2$ is Euclidean Distance: $$ \lVert x \rVert_2 = \sqrt{\sum_{i=1}^{n} x_{i}^{2}}$$

+ 推广到general distance：$\|\mathbf{x}\|_{p}=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{\frac{1}{p}} \quad(p \geq 1)$

+ 不同norm的等高线如下图所示：

  <img src="https://i.loli.net/2020/11/02/7SXo2AWGqPTx41O.png" alt="image-20201102180316414" style="zoom:50%;" />



#### 2.2. Inner Product of Vector 向量的内积

**概念**

向量的内积是一个很重要的概念，它其实涵盖了向量间的距离，向量的投影，2个向量间的角度等等重要的元素。是SVM，PCA的重要理论基础。

为了更好的理解，我们先回顾一下向量的点积（dot product）:
$$
x\top y = \sum_{i=1}^{n}  x_i y_i = \lVert x \rVert \lVert y \rVert cos\alpha
$$
其中，$\alpha$是向量$x$,$y$的角度。

其中，$\alpha$是向量$x$,$y$的角度。

向量的点积其实是一种特殊的向量的内积。严格来说，向量的内积是一个很general的概念。假设$V$是向量空间，那么任何一个映射函数$\beta$，把$V \times V$映射到一个实数$R$, 而且$\beta$满足以下性质，我们就把$\beta$称之为向量的一个内积。
+ 性质1 symmetric, $\beta(x, y) = \beta(y, x)$
+ 性质2 positive definite, $\forall x \in V \backslash \{ 0 \}: \beta(x, x) > 0$
+ 性质3 blinear: 
$$\beta(\lambda x + y, z) = \lambda \beta(x, z) + \beta(y, z)$$
$$\beta(x, \lambda y + z) = \lambda \beta(x, y) + \beta(x, z)$$

Inner Product 一般用$\langle x, y \rangle$来表示。它满足Cauchy-Schwarz不等式：

$$
\lvert \langle x, y \rangle \rvert \leq \lVert x \rVert \lVert y \rVert
$$
其中，$\lVert x \rVert = \sqrt{\langle x, x \rangle}$ 表示**向量的长度**。其实这里的向量我们是指n维空间的一个点，而向量的长度就是指这个点到原点的距离，这个距离是通过向量的内积定义的。



**向量的距离**

类似的，2个向量$x$和$y$的距离，也就是2个点的距离，也就是向量$x-y$的长度，我们在这里用$d(x, y)$表示：
$$
d(x, y) = \lVert x - y \rVert = \sqrt{\langle x - y, x - y \rangle}​
$$
$d(x, y)$被称为metric, 满足以下3个性质。

+ positive definite $d(x, y) > 0$，当且仅当 $x = y$ 时，$d(x, y) = 0$
+ symetric $d(x, y) = d(y, x)$
+ triangular inequality $d(x, z) \leq d(x, y) + d(y, z)$



**向量的角度**

由Cauchy-Schwarz不等式可得：
$$
-1 \leq \frac{\langle x, y \rangle}{\lVert x \rVert \lVert y \rVert} \leq 1
$$
所以，存在一个$\omega \in [ 0, \pi ]$使得：
$$
cos \omega = \frac{\langle x, y \rangle}{\lVert x \rVert \lVert y \rVert}
$$
这个$\omega$就是2个向量间的夹角。

如果$x \bot y$, $\langle x, y \rangle = 0$。必须要注意的一点是，我们说2个向量的垂直，是相对于在特定的内积来说的。如果是点积的话，那么垂直就是在欧式空间的垂直。



**向量的投影（重要）**

高维度的数据非常难以处理和分析，因此在面对高维数据的时候，我们首先要做的就是把数据进行压缩和降维，就是用更少的维度来表示数据。虽然这个压缩的过程中我们会损失数据的部分信息，但是在高维数据的众多维度当中，往往只有几个维度的信息是关键的。所以，我们必须找到最有价值的几个维度，然后把数据投影到那些维度上，然后在这些维度上进行数据分析，从而在精简数据的同时减少有效信息的损失。



我们下面来看看如何把高维向量$X \in R^n$投射到子空间$U \subseteq R^m$当中，其中$dim(U) = m$。$m$代表的是构成子空间$U$的基向量的个数。$m=1$,代表这个空间是一条直线；$m=2$,代表这个空间是一个平面；$m=3$,代表这个空间是一个三维空间，等等。



假设$P=(p_1,p_2,...,p_m)^\top$是高维子空间$U$的一组基向量，$p_i \in R^n$。$Y$是$x$在$U$上的投影，则$Y = PX$，矩阵表示如下：
$$
\left(\begin{array}{c}
p_{1} \\
p_{2} \\
\vdots \\
p_{m}
\end{array}\right)\left(\begin{array}{llll}
a_{1} & a_{2} & \cdots & a_{k}
\end{array}\right)=\left(\begin{array}{cccc}
p_{1} a_{1} & p_{1} a_{2} & \cdots & p_{1} a_{k} \\
p_{2} a_{1} & p_{2} a_{2} & \cdots & p_{2} a_{k} \\
\vdots & \vdots & \ddots & \vdots \\
p_{m} a_{1} & p_{m} a_{2} & \cdots & p_{m} a_{k}
\end{array}\right)
$$
其中$p_i \in R^{n}$是一个行向量，表示子空间$U$的第$i$个基；$a_j \in R^{n}$是一个列向量，表示第$j$个原始数据记录。经过变换后，原始数据$X \in \mathbb{R}^{n * k}$从$n$维降到$Y \in \mathbb{R}^{m * k}$的$m$维。也就是说，变换后的数据维度，取决于基向量的个数。



重要结论：虽然$Y$在一个$m$维的子空间里面$U \subseteq R^m$，但是它其实是一个$n$维向量。在给定$U$的一组基向量，我们只需$P=(p_1,p_2,...,p_m)^\top$, $m$个向量，就可以近似描述一个n维向量$X$。



可以看X选择不同的基向量，我们将得到不同的数据表示。至于怎么选择这组基向量，这就是PCA要探索的东西了。







#### 2.3. 矩阵行列式

行列式（Determinant）是数学中的一个函数，将一个 $ n\times n $的矩阵 $A$映射到一个标量，记作$ \det(A)$或 $\lvert A \rvert$。行列式可以看做是有向面积或体积的概念在一般的欧几里得空间中的推广。例如，在二维和三维空间中，行列式被解释为向量形成的图形的面积（如下图）或体积。矩阵$A = n \times n$的行列式为零当且仅当$dim(A) < n$。比如说在三维空间中，行列式为零意味着三个向量共线或者共面。

![image-20201102214601387](https://i.loli.net/2020/11/02/5soqOAMXZGe8uQr.png)



#### 2.4. Eigenvector and Eigenvalue 特征向量和特征值

从定义出发，$Ax=\lambda x$：$A$为$n \times n$矩阵，$\lambda$为特征值，$x$为特征向量。矩阵$A$乘以$x$表示，对向量$x$进行一次转换（旋转或拉伸）（是一种线性转换），而该转换的效果为常数$\lambda$乘以向量$x$（即只进行拉伸）。

求出特征值和特征向量有什么好处呢？ 就是我们可以将矩阵$A$特征分解。如果我们求出了矩阵$A$的$n$个特征值$\lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$,以及这n个特征值所对应的特征向量${w_1, w_2,..., w_n}$，如果这$n$个特征向量线性无关，那么矩阵$A$就可以用下式的特征分解表示：$A=WΣW^{−1}$。

其中$W$是这$n$个特征向量所张成的$n \times n$维矩阵，而$\Sigma$为这$n$个特征值为主对角线的n×n维矩阵。
一般我们会把$W$的这$n$个特征向量标准化，即满足$\lVert wi \rVert_2=1$, 或者说$w^\top_i w_i=1$，此时$W$的$n$个特征向量为标准正交基，满足$W^\top W=I$，即$W^\top=W^{−1}$, 也就是说W为酉矩阵。这样我们的特征分解表达式可以写成：$A=W \Sigma W^\top$。

要进行特征分解，矩阵A必须为方阵，而且A必须有n个线性无关的特征向量。那么如果A不是方阵，即行和列不相同时，我们还可以对矩阵进行分解吗？答案是可以，此时我们的SVD登场了，它可以对所有实数矩阵进行分解。



#### 2.5. Singular Value Decomposition 奇异值分解

**概述**

SVD是NLP里面非常重要的一个模型。SVD也是对矩阵进行分解，但是和特征分解不同，SVD并不要求要分解的矩阵为方阵。假设我们的矩阵$A$是一个$m \times n$的矩阵，那么我们定义矩阵$A$的SVD为：$A=U \Sigma V^\top$，其中$U$是一个$m \times m$的矩阵，$\Sigma$是一个$m \times n$的矩阵，除了主对角线上的元素以外全为$0$，主对角线上的每个元素都称为奇异值，$V$是一个$n \times n$的矩阵。$U$和$V$都是酉矩阵，即满足$U^\top U=I$,$V^\top V=I$。



**SVD的一些性质**

对于奇异值,它跟我们特征分解中的特征值类似，在奇异值矩阵中也是按照从大到小排列，而且奇异值的减少特别的快，在很多情况下，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。也就是说，我们也可以用最大的k个的奇异值和对应的左右奇异向量来近似描述矩阵。也就是说：
$$A_{m \times n}=U_{m \times m}\Sigma_{m \times n}V^\top_{n \times n} \approx U_{m \times k} \Sigma_{k \times k}V^\top_{k \times n}$$

其中$k$要比$n$小很多，也就是一个大的矩阵$A$可以用三个小的矩阵$U_{m \times k}$, $\Sigma_{k \times k}$, $V^\top_{k \times n}$来表示。

由于这个重要的性质，SVD可以用于PCA降维，来做数据压缩和去噪。也可以用于推荐算法，将用户和喜好对应的矩阵做特征分解，进而得到隐含的用户需求来做推荐。同时也可以用于NLP中的算法，比如潜在语义索引（LSI）。



#### 2.6. Vector to Tensor

简单来说，Tensor就是高维数组。

![2](https://i.loli.net/2020/11/02/UKfYwuCO1IjMcRg.png)

#### 2.7. Jacobian and Hessian Matrices

假设$F: R_n→R_m$是一个从欧式n维空间转换到欧式m维空间的函数. 这个函数由m个实函数组成: $y_1(x_1,…,x_n), …, y_m(x_1,…,x_n)$. 这些函数的偏导数(如果存在)可以组成一个m行n列的矩阵, 这就是所谓的雅可比矩阵。
$$
\mathbf{J}_{f}=\left[\begin{array}{ccc}
\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{1}}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{m}}{\partial x_{1}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}}
\end{array}\right] \quad \text { i.e. } \quad\left[\mathbf{J}_{f}\right]_{i j}=\frac{\partial f_{i}}{\partial x_{j}}
$$


海森矩阵(Hessian matrix或Hessian)是一个自变量为向量的实值函数的二阶偏导数组成的方块矩阵, 此函数为$f(x_1,x_2…,x_n)$。
$$
\nabla^{2} f=\left[\begin{array}{ccc}
\frac{\partial^{2} f}{\partial x_{1}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1} \partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial^{2} f}{\partial x_{n} \partial x_{1}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}}
\end{array}\right] \quad \text { i.e. } \quad\left[\nabla^{2} f\right]_{i j}=\frac{\partial^{2} f}{\partial x_{i} \partial x_{j}}
$$


### 3. Convex Set and Functions 

凸函数具有很好的数学性质，容易求导，容易梯度优化。如果可以判断一个函数是凸函数，那么就可以直接套用凸函数的数学性质。



#### 3.1. 定义

A function is convex if 
$$
f(t \mathbf{x}+(1-t) \mathbf{y}) \leq t f(\mathbf{x})+(1-t) f(\mathbf{y})
$$
for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$ and all $t \in[0,1]$。



#### 3.2. Jenson不等式

如果$f$是凸函数，那么有
$$
f(\frac{x+y}{2}) \leq \frac{f(x)+f(y)}{2}
$$
这是Jensen不等式的常规形式，还可以扩展为：如果$x_1, x_2, ..., x_k \in dom(f)$，$\theta_1, \theta_2, ..., \theta_k \geq 0$ 而且 $\theta_1 + \theta_2 + ... + \theta_k = 1$，则下式成立。

$$
f(\theta_1x_1 + ... + \theta_kx_k) \leq \theta_1f(x_1) + ... + \theta_kf(x_k)
$$
扩展至概率形式则为，把$\theta_1, \theta_2, ..., \theta_k$看成是概率分布$p_1, p_2, ..., p_k$，则有
$$
f(p_1x_1 + ... + p_kx_k) \leq p_1f(x_1) + ... + p_kf(x_k)
$$
写成期望形式为：
$$
f(Ex) \leq Ef(x)​
$$


### 4. Probability and Statistics