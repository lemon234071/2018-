# 扫盲贴  

经验贴

	陈成龙经验（腾讯主办方大神，kaggle中国区第一科学家，此处特别推荐腾讯比赛的公众号，一般你参加什么比赛都应该去关注他官方发布消息的地方，特别是QQ群，里面都是大佬。）
	https://mp.weixin.qq.com/s/BE1mfmKJTsDSwWi16mllNA
	如何在 Kaggle 首战中进入前 10%
	https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/
	第一届比赛代码
	https://github.com/BladeCoda/
	有关stacking
	https://blog.csdn.net/ben3ben/article/details/74838338
	
	场景相关相关
	广点通：https://www.csdn.net/article/2015-08-27/2825557
	阿里妈妈：https://blog.csdn.net/seu_yang/article/details/51966385
数据集  

特征分析  
	
	
	# 数据可视化
	https://www.kaggle.com/benhamner/python-data-visualizations
	http://python.jobbole.com/87136/
	# 简单分析
	https://blog.csdn.net/u010289316/article/details/51571540
	#数据清洗
	http://baijiahao.baidu.com/s?id=1581755863609513980&wfr=spider&for=pc
	# #########特征工程######### #
	https://www.zhihu.com/question/29316149
	https://blog.csdn.net/sqiu_11/article/details/59487177
	http://www.cnblogs.com/jasonfreak/p/5448385.html  
	http://www.cnblogs.com/jasonfreak/p/5619260.html  
	http://www.cnblogs.com/jasonfreak/p/5448462.html 
	#结合Scikit-learn介绍几种常用的特征选择方法
	https://www.cnblogs.com/hhh5460/p/5186226.html
	https://blog.csdn.net/u013421629/article/details/78769949
	#冷启动特征
    https://www.cnblogs.com/nolonely/p/6863574.html
    #连续特征离散化
    https://blog.csdn.net/u013818406/article/details/70494800
	#one-hot and labelEncoder
	https://blog.csdn.net/accumulate_zhang/article/details/78510571
	# GDBT叶子特征
	https://blog.csdn.net/shine19930820/article/details/71713680
	# 特征组合，这里的组合变化远不限于把已有的特征加减乘除（比如Kernel Tricks之类）
	#杂七杂八
	http://www.docin.com/p-108257148.html
	https://www.douban.com/note/642960921/
	#节省内存
	https://www.cnblogs.com/xxr1/p/7397767.html
	https://blog.csdn.net/u012347642/article/details/78555132
	https://blog.csdn.net/weiyongle1996/article/details/78498603?locationNum=3&fps=1
	http://www.360doc.com/content/17/0305/18/1489589_634208133.shtml
	https://www.zhihu.com/question/50477642
embedding

	https://developers.google.cn/machine-learning/crash-course/embeddings/obtaining-embeddings  

调参  

	#能凭经验就凭经验吧，hyperopt这种调参库也很慢的，格子就算了
	https://www.zhihu.com/question/34470160?sort=created  
lgb  
	https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst  
	https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst  
	https://zhuanlan.zhihu.com/p/27916208  
	https://blog.csdn.net/weiyongle1996/article/details/78446244  
	https://sites.google.com/view/lauraepp/parameters  
	http://lightgbm.apachecn.org/cn/latest/Python-API.html#scikit-learn-api  
  
一些聊天记录：

    如果你发现了两组强特，他们相关性很强，可以尝试用他们分别训练模型然后融合，可以得到不错的效果。（理论上）

	基础特征：计数特征、转化率、比例特征等各种基本的特征；

	特征提取方式有以下几个方面考虑：

	基于cv统计、贝叶斯平滑等方法，能够很好的修正线上线下的特征分布不一致的问题；
	  
    #w2v
    两个vec的距离越近，这两个人的兴趣就越接近,太稀疏w2v训不好,对特征使用w2c虽然解决了稀疏问题，但是使用太多会使模型学习变得困难。这个要折中。  
    其实可以对照着各个特征的describe，有些特征不需要w2v感觉,单个进程要很久很久的，提醒一下,反正我是interest  kw topic  和另外的分开的thread,内存小的话吧size设小点，小数点后面保留小点  
    http://d0evi1.com/gensim/  
    
    #coo_matrix??????sparse matrix,csr,csc,  
    https://blog.csdn.net/u013010889/article/details/53305595  

