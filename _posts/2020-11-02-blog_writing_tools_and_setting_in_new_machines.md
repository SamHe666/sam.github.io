---
layout: post
title: "在新电脑上配置写博客的工具"
date:   2020-11-02
tags: [备忘]
toc: true
author: Sam
---

### 下载Github客户端

Github下载地址：https://desktop.github.com/

1. 登录Github账号：QQ邮箱
2. Clone Repository到本地：samhe666.github.io
3. 博客文章用用markdown语法，写好统一放在_post文件夹下上传。



### 下载Typora和Picgo

Typora下载地址：https://typora.io/

Picgo下载地址：https://github.com/Molunerfinn/PicGo/releases



### 配置Typora + Picgo

1. Gitee的图床已经设置好了，这里并不需要去管它。地址：https://gitee.com/samhe666

2. Picgo->图床设置->Github图床：

   设定仓库名：SamHe666/blog-imgs/

   设定分支名：master

   设定Token：192ed2f677afecc211d321d5b41912ff

   指定存储路径：留空

   设定自定义域名：https://raw.githubusercontent.com/SamHe666/blog-imgs/master

   

   **备注：如果忘了Token，然后就去到Gitee的仓库里面，SamHe666->setting->Security Settings->Personal access tokens->删除并生成新的token即可。**

3. Picgo->PicGo设置->设置Server->参照如下：

   ![image-20201102122926605](https://i.loli.net/2020/11/02/wc8KMpjRBmCl9HO.png)

4. Picgo->PicGo设置->时间戳重命名->打开。

5. Typora要先设置成中文，picgo.app只支持简体中文。然后偏好设置->图像->按照以下配置->验证图片上传选项。

   ![image-20201102122314838](https://i.loli.net/2020/11/02/UG3jxoaAlEiIQXK.png)

6. 然后无论是截图粘贴还是拖拉插入图片，图片都会实现自动上传并获取url。



### 写博客

1. 博客文章必须按照统一的命名格式 `yyyy-mm-dd-blogName.md`
2. 开头要贴配置信息段（YAML Front Matter）。新建文件，打---然后回车即可。
3. 在文章开头信息中心增加 `toc: true` 描述即可打开文章目录显示。



### Reference

1. https://www.jianshu.com/p/a1e2cf01e05f
2. https://www.jianshu.com/p/4cd14d4ceb1d