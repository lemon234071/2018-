此贴代码分析进行到模仿top3 attention机制,持续更新ing.....

2018腾讯算法大赛第29/1500名，此贴总结TOP10代码，初次大数据（机器学习）比赛，扫盲贴。此次TOP1由葛文强团队的自创的NN取得，模型类似于其IJCAI2018模型图片位于：  

	./picture/top1.png
第三名为清华大学茶园和贵系大佬自创的DEEP INTEREST FFM，PPT海报位于：  

	./picture/top3.pdf
其余大佬模型多基于郭大和渣大分享的NFFM模型稍作改动，此模型为2017年第一届比赛南大熊猫冠军模型，github地址：  

	郭大https://github.com/guoday/Tencent2018_Lookalike_Rank7th（郭大的Xdeepfm听说有效果，还未来得及研究）
	渣大https://github.com/nzc/tencent-contest
top20经验贴传送门：  
	
	官方baseline:
	https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline

	Tensorflow:[2018腾讯广告算法大赛总结（Rank6）-模型篇](https://zhuanlan.zhihu.com/p/38443751)
	Pytourch:[第二届腾讯广告算法大赛总结（Rank 9）](https://zhuanlan.zhihu.com/p/38499275?utm_source=com.tencent.tim&utm_medium=social&utm_oi=571282483765710848)
	[2018腾讯广告算法大赛总结/0.772229/Rank11](https://zhuanlan.zhihu.com/p/38034501)
以上模型皆属于neutral network模型，因当数据量很大，机器限制又很高的时候，NN模型很适应场景，ctr nn模型扫盲贴在后面，此次数据量过大，复赛对GPU和内存要求很高，大概400G内存，我将在下面的目录下放一个我赛后学习测试用的数据集，16G笔记本可跑：  
	
	./datas/

赛题详情:  

	./2018腾讯算法大赛参赛手册  

大数据（机器学习）比赛大致流程如下，旨在预测在官方给的计量方法上取得高分（如AUC，logloss）：  
	
	1、做特征
	2、喂模型
	3、融合

相关基础知识扫盲帖：  

	./information/

特征----本次比赛大家特征都大同小异：  

	常见几种特征提取代码./feature.py  
	怎么挖掘特征？渣大传送门https://zhuanlan.zhihu.com/p/38341881

PS:具体情况具体分析，有时候选特征可以看LGB提供的API看重要度，有时候需观察与label的相关度，但是要注意不要leak（即用未来的数据来预测过去，用label去预测label，会造成线下很高，线上很低），可以用KFOLD来避免。但是这些都不保证一个特征有用无用，最靠谱的还是直接放进去看线上是否提高了，但是这样你的时间不够的，复赛又可能有别的强特。另外学习怎么做特征，最好的办法就是看大赛TOP10的PPT和代码，看太多特征工程原理上的东西会浪费你很多时间。另外，基本的可视化分析，这个东西扫盲贴里有介绍，但是赛后发现大佬们分析的东西和我这样的新手菜鸟分析的东西完全不是一个东西啊。所以尚在摸索中。 

模型----本次比赛为CTR类比赛，常用模型有lgb,xgb,catboost等树类模型，FFM模型，还有一系列复杂的NN深度模型.  

融合----我用的加权融合，台大三傻的反sigmoid融合然后排列组合  
	
	大概就是利用哈夫曼原理，进行哈夫曼融合，把每个单次线上成绩进行哈夫曼反sigmoid，穿插进行加权融合，因为有的时候加权比反sigmoid的效果好，具体慢慢详细研究介绍
	代码：./*********.py

初赛：  
初赛精力主要要放在做特征上，微软的lgb模型效率很高，可以尽可能的去做特征然后愉快的喂给LGB即可。  

复赛：   
我们重点说说复赛，上面说了复赛数据量和机器限制，一开始大家考虑deepfm模型，后来郭大渣大开源NFFM，所以开始入手研究NN模型，下面是CTR的NN扫盲贴两篇：  

	渣大知乎：https://zhuanlan.zhihu.com/p/32885978
	石晓文大神简书：https://www.jianshu.com/nb/21403842
	辛俊波知乎：https://zhuanlan.zhihu.com/p/35484389
	
简单来说deepFFM是将deep层和ffm层wide方向concat了，nffm是在串行将FFM层的结果再喂给DENSE层，关于其效果优良的可解释性至今未查清，一老师给出的是，不同的环境和场景，模型效果也可能不同，至于TOP3的interest网络，attention机制的可解释性就比较好理解了，本次比赛中的多值特征，大家都去了average pool(类似max pool 你懂得），但是对不同广告，你兴趣权重肯定是不一样的，比如你点击口红广告，不可能跟你喜欢的AJ款式有很大权重，所以这里用了ATTENTIONG机制动态分配权重，这个网络其实就是lr表达一阶特征，FFM表达二阶特征，TOP1团队则用MVM取了个log表达了更高阶特征，其余细节，可以看论文中讲述：  

	阿里巴巴din网络https://arxiv.org/pdf/1706.06978.pdf

本文代码起初借鉴于初赛不知名大佬的MXNET新框架gluon，apache的动静集合的框架其实我很喜欢，但是有些API还是维护的比较慢的，所以正转战TF，和torch，哎，很桑心，正在尝试TOP1模型复现，以及spark进行数据处理，最后会总结出一个比较方便的通用框架持续更新ing...
	

另外一些踩过的坑与大家分享：  

	# 特征清洗很重要，不然要重做特征很多次，做大量特征前要小批量验证代码正确，不然有BUG就很浪费时间
	#内存不够的时候read_csv数据类型别用默认的64位
	#del 后的内存要和谐的collect()一下
