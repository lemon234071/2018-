http://www.cnblogs.com/jasonfreak/p/5448385.html  
http://www.cnblogs.com/jasonfreak/p/5619260.html  
http://www.cnblogs.com/jasonfreak/p/5448462.html  

first:  

data.analysis：    
1,df.descirbe  
2,plot  

warning here cross validate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(and know some pickle  
3,coder(w2v,one-hot,cv,lablecoder,meancoder....  


http://www.dataguru.cn/article-12877-1.html  
http://www.docin.com/p-1144110216.html  
https://blog.csdn.net/shichaog/article/details/72848502  
https://www.jianshu.com/p/b26a5ba91940
https://www.zhihu.com/question/19971859

4,corr()  
5,wash data  
6,trick feature  

then pass the code then filter







note:  
samples: +/- --> 1:20  


dalao`s words:  

feature:  

    我将每个兴趣里面的 个数统计了下，，蛮有效的,kw2这个特征特别有用啊……uid这个统计特征会leak,其实可以用现有标签去推断婚姻关系  
    地理位置可以按照城市相似性比如一线城市二线城市之类的分桶离散化啊  
    
    凡是不能比较大小的都需要one hot。有一些可以比较大小的也可以离散化之后one hot。当然也可以labelencoder  
    。。直接onehot好像噪声还是太大了点,  
    比如 你有100000个商品id 你用 countvector 就不合适了。  
    https://developers.google.cn/machine-learning/crash-course/embeddings/obtaining-embeddings  
    两个vec的距离越近，这两个人的兴趣就越接近,太稀疏w2v训不好,对特征使用w2c虽然解决了稀疏问题，但是使用太多会使模型学习变得困难。这个要折中。  
    其实可以对照着各个特征的describe，有些特征不需要w2v感觉,单个进程要很久很久的，提醒一下,反正我是interest  kw topic  和另外的分开的thread,内存小的话吧size设小点，小数点后面保留小点  
    http://d0evi1.com/gensim/  
    
    coo_matrix??????sparse matrix,csr,csc,  
    https://blog.csdn.net/u013010889/article/details/53305595  
    做一个str拼接后映射为  
    
    CountVectorizer在统计词频时，会把长度为1的词都过滤掉？stopword我看了，并没有这些词,。空格也算stopwords吧  
    
    一开始我准备 直接用余弦相似来做的  
    
    正例训练一个特征 向量 负的 训练一个特征向量  
    
    结果 哪些文本特征不知道怎么处理,embedding然后取平均  
    
    用户特征集中出现但不在训练集中的用户特征需要处理。用户特征中出现重复用户，不同特征了吗？  
    
    热门的物品会在各种特征的用户中都具有比较高的权重，给用户推荐热门物品并不是推荐系统的主要任务，推荐系统应该帮助用户发现他们不容易发现的物品。因此，我们可以把 这里写图片描述定义为喜欢物品i的用户中具有特征f的比例：  

这里分母中的 这里写图片描述的目的是解决数据稀疏的问题。如果有一个用户只被1个用户喜欢过，而这个用户刚好有特征f，那么p(f,i)=1.但是这种情况并没有统计意义。我们为分母加上一个比较大的数字，可以避免对这样的物品产生比较大的权重。  
parameters:  

    嗯嗯，就把num_leavse调高了点,主要调size和min_count吧  

nn model:  

    用NN加流式读取做的,因为NN的batch 训练的啊，如果batch_size是64，那每次读64行就行了,加个max_feature＝200，就快很多了,不是lgb，是countvector,  
    不过FTRL可提升的空间不大,我的网络用的是NLP的embedding加一些pooling，最后MLP    
    /笑哭 nn输出＋fm+lr+stacking + belding + stacking + blediing  
    http://url.cn/5ITzZib


    https://zhuanlan.zhihu.com/p/32885978?utm_medium=social&utm_member=MTQyNzc1ZGNiYmVjMDVkZGM2MTgwZmUwOTljOWNhNmY%3D&utm_source=qq

    nzc.(120446477) 15:06:42
    比如w2v这种
    Hermite(910988450) 15:06:45
    先选部分数据跑
    Hermite(910988450) 15:06:58
    然后再跑整个数据 就会快了
    nzc.(120446477) 15:07:07
    基于dnn得到的有监督的embeddibg
    nzc.(120446477) 15:07:27
    比w2v这种无监督得到的好多了
    nzc.(120446477) 15:07:45
    所以这种特征如果是dnn的话，根本不用做
    @李乐乐乐乐啊🤵🏻(1137440039) 15:08:27
    嗯，embeding效果也不错
    Trust the Process-中山大学(314123799) 15:08:32
    @nzc. 你embedding做了几维啊
    nzc.(120446477) 15:09:14
    4-10就够了啊
    nzc.(120446477) 15:09:24
    不用太大的吧

    Trust the Process-中山大学(314123799) 15:10:27
    之前有人跟我说取log2差不多
    Hermite(910988450) 15:10:52
    log2是怎么来的
    nzc.(120446477) 15:11:05
    看结构的吧
    nzc.(120446477) 15:11:20
    有些网络结构要求等长
    Trust the Process-中山大学(314123799) 15:11:33
    做内积就要等长咯
    nzc.(120446477) 15:11:38
    dcn这种好像是根号2




    Trust the Process-中山大学(314123799) 15:12:18
    试了deepfm，效果不咋地

    bupt-划水(946533269) 15:41:48
    多值离散特征你在deepfm怎么处理的@Trust the Process-中山大学
    15:42:20
    旗开得胜siiussilen撤回了一条消息
    Trust the Process-中山大学(314123799) 15:42:34
    embedding取均值
others:  

  自己做的那叫验证集。而且也是看验证集的AUC来调模型，  
  每次读入部分列或部分行，pd.readcsv 的chunksize了解一下,  
  昨晚参考了郭大的意见，，，用了ftrl,kbqjf就行了  
  indicator 和embedding  
