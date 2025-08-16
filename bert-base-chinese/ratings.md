### 说明
ratings.csv可以是你自己的根据业务需求制定的训练数据集，拿来微调模型。

github:
https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb

### 详情
0. **下载地址：** [百度网盘](https://pan.baidu.com/s/1SbfpZb5cm-g2LmnYV_af8Q)\n",
1. **数据概览：** 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据\n",
2. **推荐实验：** 推荐系统、情感/观点/评论 倾向性分析\n",
2. **数据来源：** [亚马逊](https://www.amazon.cn/)\n",
3. **原数据集：** [JD.com E-Commerce Data](http://yongfeng.me/dataset/)，Yongfeng Zhang 教授为 WWW 2015 会议论文而搜集的数据\n",
4. **加工处理：**\n",
    1. 将全角字符转换为半角字符，并采用 UTF-8 编码\n",
    2. 整理成与 [MovieLens](https://grouplens.org/datasets/movielens/) 兼容的格式\n",
    3. 进行脱敏操作，以保护用户隐私"