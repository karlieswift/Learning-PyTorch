# Learning-PyTorch
The main content of this project: CNN RNN LSTM....... and so on.
# RNN-LSTM code introduce
## 项目的描述：
### 数据描述：
#### 1-首先从数据网站 https://ai.stanford.edu/~amaas/data/sentiment/    下载数据
#### 2-解压数据分为train和test文件夹,每个文件夹下有pos和neg文件夹,每个文件夹各有12500个.txt文件，共50000数据

## 目的：通过rnn,lstm进行分类,数据是neg or pos

### 思路介绍：
#### 1.首先拿到的数据是字符串，我们将字符串进行处理，转换为数字进行分析，我们将这种数字和字符串之间的关系存入一个文件保存，便于下次直接使用。、
这个是Pword2sequence模块。
#### 2-既然要训练数据，我们可以定义一个dataset模块进行数据读取（包括字符处理和调用上面的Pword2sequence模块将字符串转为数字）
#### 3-对上面的数据进行训练，model模块


### 整个项目主要涉及到Pword2sequence，dataset和model1的编写
#### Pword2sequence：字符串和数字之间的对应关系。并把这种关系进行保存。而且和dataset代码之间没有特别多的必然联系(仅仅调用tokenize函数)
#### dataset：对数据进行处理和加载。(调用了Pword2sequence的transform进行字符串转数字)
#### model：训练。(调用dataset的get_dataloader进行加载数据)，其中model和Pword2sequence模块不打交道。



### Pword2sequence.py
目的：将单词序列进行序列化，对单词进行编码和译码
#### build_vocabulary()函数模块是将对dict(单词:数字编码)和inverse_dict(数字编码:单词) 这两个字典设计
具体处理：删除次数小于min的单词，删除次数大于max的单词，最多保留的词语 首先要保留最多的 进行排序
#### fit()统计词频
#### inverse_transform()将数字编码转化为对应的单词
#### transform()把句子转化为序列--把单词转化编码
最终通过模型保存把字典等属性保存
pickle.dump(ws,open('./model/ws.pkl','wb'))
以后只要重新加载就可以读取字典
ws=pickle.load(open('./model/ws.pkl','rb'))


### dataset.py的主要目的就是实现2个功能：1-对数据的预处理 2-读取数据
tokenize()特殊字符处理
collate_fn()这个地方我们会用到上面的transform() 这样将字符串转化为数字
get_dataloader()加载数据


## model1.py
进行数据训练

lib.py



# Mnist-CNN-MLP code introduce
look look code!!!!!

