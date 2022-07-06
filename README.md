# kaggle-uspppm-silver-medals


从六月以来，一直在华为实习，没抽出时间写一写文章，最近很开心的是刚结束的Kaggle比赛[U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview)中取得了银牌，很感谢队友给予的大力帮助，在这里简述一下具体的做法，以作回顾。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9ae821c803c54f83bdcbadd17f262e45.png)

# 0、比赛内容背景
背景：以美国专利库为主要数据集，匹配专利文档中的关键词短语来提取相关信息
 - 类型：深度学习/NLP
 - 数据：成对的短语（anchor和target），在0到1的范围内评估它们的相似性，训练集36473对短语，训练集大约有12000对短语
 - 评估标准：皮尔逊相关系数

# 1、数据处理
数据处理方面一些常规的处理方法，例如转换成小写、去掉前后的空格等就不再赘述。
主要说一些数据集的处理，本次数据集主要处理：

 1. 讨论区引入了CPC文件，该文件中每个专利代码的标题作为title text。
 2. 对anchor和context进行groupby，获得聚合后的targets列表。
 3. 在2的基础上生成文本 **anchor[SEP]target[SEP]title[SEP]gp_targets**。
 4. 使用groupKfold将数据拆分成训练集和验证集，能够有效地避免数据泄露导致的线上线下分数差别过大问题。

# 2、模型/结构
模型使用的是**Deberta（主流）、bert for patent、ELECTRA、Funnel-Transformer**四者的融合。

|Model| seq_length | CV| PB |
|--|--|--|--|
| deberta-v3-large | 200 | 0.844 | 0.842 |
| electra-large | 200 | 0.832 | 0.833 |
| funnel-large | 200 | 0.824 | 0.825 |
| bert for patent | 200 | 0.824 | 0.824 |
| Ensemble | | | 0.85x |

**Deberta**作为Bert的改进版本，主要包含以下三点：

 1. 首先是解耦注意机制(Disentangled attention)，每个词分别用两个向量表示，分别对其内容和位置进行编码，单词之间的注意权值分别用其内容和相对位置的解耦矩阵计算。
 2. 一种增强的掩码解码器用于在解码层中合并绝对位置，以预测模型预训练中的掩码tokens。
 3. 此外，采用一种新的对抗训练方法用于微调，以提高模型的泛化能力

Deberta与bert不同的是，输入层中的每个单词都用一个向量表示，这个向量是单词(内容)嵌入和位置嵌入的总和，而Deberta中的每个单词都用两个向量表示，分别对其内容和位置进行编码，分别根据单词的内容和相对位置，采用解耦矩阵计算单词间的注意权值。

实现方面主要利用了**Huggingface Transformer**，其能够帮我们跟踪流行的新模型，并且提供统一的代码风格来使用BERT、XLNet和GPT等等各种不同的模型。而且它有一个模型仓库，所有常见的预训练模型和不同任务上fine-tuning的模型都可以在这里方便的下载。其主要的三大类：

 - **Model 类** ：包括30+的PyTorch模型(torch.nn.Module)和对应的TensorFlow模型(tf.keras.Model)。
 - **Config 类**：它保存了模型的相关(超)参数。我们通常不需要自己来构造它。如果我们不需要进行模型的修改，那么创建模型时会自动使用对于的配置。
 - **Tokenizer类**：它保存了词典等信息并且实现了把字符串变成ID序列的功能。

上述模型作为backbone，最后加上一个全连接层与sigmod层

# 3、其他方法

 - 损失函数：MSELoss
 - 优化器：AdamW
 - 调度器：CosineAnnealingWarmRestarts
 - FGM对抗训练（帮助不大，上分很微弱）

# 4、总结
**多关注数据方面的处理**，数据的处理往往是最重要的！！！！
多进行模型的融合，比无目的的调参要有用一些！！！！
