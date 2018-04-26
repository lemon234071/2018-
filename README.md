# 2018spa  
qq2018spa  
cut为切分训练集和用户特征集  
merge文件更新    
statistic为分片统计  



经验贴

	https://github.com/ChenglongChen/Kaggle_HomeDepot
	https://github.com/BladeCoda/Tencent2017_Final_Coda_Allegro
	https://github.com/jiaqiangbandongg/RPTune
	https://github.com/ColaDrill/tc_comptition
	https://blog.csdn.net/nzcusing/article/details/73864675
	https://blog.csdn.net/ben3ben/article/details/74838338
	
	相关
	广点通：https://www.csdn.net/article/2015-08-27/2825557
	阿里妈妈：https://blog.csdn.net/seu_yang/article/details/51966385
数据集  

特征分析  
	
	#统计lable在不同的特征类型下的转化率（！！！！！！！！注意这里若结合不同广告的情况可能观测出良好效果
	# if you want vision turn to https://www.kaggle.com/benhamner/python-data-visualizations
	# 特征初选（！！！！！！！注意需要特征清洗
	#https://blog.csdn.net/u013421629/article/details/78769949
	# http://www.docin.com/p-108257148.html
	# https://blog.csdn.net/sqiu_11/article/details/59487177
	# https://www.cnblogs.com/hhh5460/p/5186226.html
	# 构造新特征https://blog.csdn.net/shine19930820/article/details/71713680
	# 特征组合，这里的组合变化远不限于把已有的特征加减乘除（比如Kernel Tricks之类）
	其他
	https://blog.csdn.net/u010289316/article/details/51571540
	https://www.douban.com/note/642960921/
	https://blog.csdn.net/u010289316/article/details/51571540
可视化  
	
	http://python.jobbole.com/87136/
数据过处理办法  

	数据清洗http://baijiahao.baidu.com/s?id=1581755863609513980&wfr=spider&for=pc
	https://www.cnblogs.com/xxr1/p/7397767.html
	https://blog.csdn.net/u012347642/article/details/78555132
	https://blog.csdn.net/weiyongle1996/article/details/78498603?locationNum=3&fps=1
	https://www.v2ex.com/t/272090
	http://www.360doc.com/content/17/0305/18/1489589_634208133.shtml
	https://www.zhihu.com/question/50477642
	https://www.aliyun.com/jiaocheng/520887.html

经验贴  

特征工程  

	基本知识
	http://app.myzaker.com/news/article.php?pk=5a1153cbd1f149e13b000080
	特征工程与推荐系统
    https://www.zhihu.com/question/29316149
    冷启动特征
    https://www.cnblogs.com/nolonely/p/6863574.html
    连续特征离散化
    https://blog.csdn.net/u013818406/article/details/70494800
    推荐系统
    https://blog.csdn.net/taoyanqi8932/article/details/62052684
    三种文本特征提取（TF-IDF/Word2Vec/CountVectorizer）及Spark MLlib调用实例（Scala/Java/python）
    https://blog.csdn.net/liulingyuan6/article/details/53390949
	one-hot and label
	https://blog.csdn.net/accumulate_zhang/article/details/78510571
	平滑降维等
	http://blog.sina.com.cn/s/blog_66a6172c0102v3eq.html
调参  

	https://www.zhihu.com/question/34470160?sort=created  
lgb  

	https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst  
	https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst  
	https://zhuanlan.zhihu.com/p/27916208  
	https://blog.csdn.net/weiyongle1996/article/details/78446244  
	https://sites.google.com/view/lauraepp/parameters  
	http://lightgbm.apachecn.org/cn/latest/Python-API.html#scikit-learn-api  


  思路   （部分官方群里发现的trick陆续整理）：  

	热门的物品会在各种特征的用户中都具有比较高的权重，给用户推荐热门物品并不是推荐系统的主要任务，推荐系统应该帮助用户发现他们不容易发现的物品。因此，我们可以把 这里写图片描述定义为喜欢物品i的用户中具有特征f的比例：  

	这里分母中的 这里写图片描述的目的是解决数据稀疏的问题。如果有一个用户只被1个用户喜欢过，而这个用户刚好有特征f，那么p(f,i)=1.但是这种情况并没有统计意义。我们为分母加上一个比较大的数字，可以避免对这样的物品产生比较大的权重。   
可处理处：  
	
	用户特征集中出现但不在训练集中的用户特征需要处理。
	用户特征中出现重复用户，不同特征了吗？


PS:
    one-hot,word2vec,贝叶斯平滑,cv,滑窗,stacking

    1、删除线上线下均值差异30%以上的特征；2、通过xgboost计算的特征重要性，删除重要性较低的特征；3、通过wrapper的方式选择特征。通过以上方式能够保证线上线下的特征稳定，但这工作在决赛数据量大的情况下会比较耗时。
在决赛阶段，每加入一部分特征，通过线上的成绩反馈来选择特征的去留。

    另外，如果你发现了两组强特，他们相关性很强，可以尝试用他们分别训练模型然后融合，可以得到不错的效果。（理论上）

    最后我的融合方案是：

    用xgboost来stacking训练lightGBM(Model1)

    用lightGBM来stacking训练xgboost(Model2)

    训练单模型lightGBM(Model3)

    结果加权：0.25*Model1+0.25*Model3+0.5*Model2

    特征的提取主要有以下几个方面：

基础特征：计数特征、转化率、比例特征等各种基本的特征；
线上的特征：基于当天数据统计的用户行为、app行为的特征；
用户行为挖掘特征：word2vec计算用户行为与历史行为的关联；
特征提取方式有以下几个方面考虑：

基于cv统计、贝叶斯平滑等方法，能够很好的修正线上线下的特征分布不一致的问题；
特征提取主要有基于全局的数据统计以及滑窗的历史统计。
基于全集的数据统计生成的特征：是决赛中主要的特征提取方式，效果比较平稳，而且信息量比较多，但容易会有信息泄露的问题需要通过cv统计来避免，而且难以反映时间变化的信息。
基于滑窗的生成特征：能反映时序上的信息，不会有信息泄露的问题。但是生成的特征数量多，线上线下的分布差异比较大，特征工程方面的工作量比较大。
因此，比赛中我选择了两种生成特征的方式来产生不同的模型进行融合。   
