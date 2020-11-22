---
layout: post
title: "NLP-秦曾昌 Chapter 7"
date:   2020-11-22
tags: [NLP]
toc: true
author: Sam
---

## 语言技术 - 主题模型

### 1. 概率图模型

虽然今天用深度学习比较多，但是对于小文本、数据量不是特别大的情况，graphic model还是能提供很多有用的思路去解决问题。



其实概率图模型不是什么新东西，其实就是用图的形式来表达一个概率模型。就算是一个简单的条件概率或者是较为复杂的隐马尔科夫链模型也都可以用图来表示。图的表示方法可以让你更好的理解整个模型，比如说直观地看到变量的依赖关系啊，一个变量由什么变量决定的，那些是观测变量，那些是隐含变量等等。



### 2. 主题模型 

#### 2.1. LSA

LSA: Latent Semantic Analysis 潜在语义分析。



#### 2.2. PLSA 

PLSA: Probabilistic Latent Semantic Analysis，说白了就是SVD分解。

