"""
 Env: /anaconda3/python3.7
 Time: 2021/7/24 9:00
 Author: karlieswfit
 File: Pword2sequence.py
 Describe: 实现字符到数字的转换Word2Squence
"""
class Word2Squence:
    UNK_TAG="UNK"  #没有的单词
    PAD_TAG="PAD"  #填充的单词
    UNK=0
    PAD=1
    def __init__(self):
        self.dict={  #存储单词对应的数字 即给一个单词我们就能查到该单词对应的数字编码
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.inverse_dict={} #与self.dict相反 给一个数字编码就能知道它对应的单词

        self.count={}  #词频 统计这个单词出现的次数

    def fit(self,sentence):
        """
        :des:统计词频
        :param sentence: 是一个list
        """
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1

    def build_vocabulary(self,min=0,max=None,max_feature=None):
        """
        对self.dict和inverse_dict 进行设计
        :param min: 过滤最少的单词出现的次数
        :param max: 过滤最多的单词出现的次数
        :param max_feature: 最多保留的词语
        :return:
        """
        #删除次数小于min的单词
        if min is not None:
            self.count={word:count for word,count in self.count.items() if count>min}
        # 删除次数大于max的单词
        if max is not None:
            self.count = {word: count for word, count in self.count.items() if count < max}
        #最多保留的词语 首先要保留最多的 进行排序
        if max_feature is not None:
            self.count=dict(sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_feature])

        # todo 对count的单词进行编码 即每个单词对应一个自己的数字编码 存入 self.dict
        for word in self.count:
            self.dict[word]=len(self.dict) #通过dict的大小进行升序编码  self.count的单词不可能重复


        # todo self.inverse_dict 给一个数字编码就能知道它对应的单词
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))
        # for word in self.dict:
        #     self.inverse_dict[dict[word]]=word

    def transform(self,sentence,max_len=None):
        """
        把句子转化为序列--把单词转化编码
        :param sentence: list 单词
        :param max_len: 对句子进行裁剪和填充
        :return:
        """
        if max_len is not None:
            if max_len>len(sentence): #进行填充
                sentence=sentence+[self.PAD_TAG]*(max_len-len(sentence))
            else: #进行裁剪
                sentence=sentence[:max_len]

        list=[]
        for word in sentence:
            list.append(self.dict.get(word,self.UNK))
        return list

    def inverse_transform(self,index):
        """
        将数字编码转化为对应的单词
        :param index: list 的数字编码
        :return:
        """
        list=[]
        for i in index:
            list.append(self.inverse_dict.get(i))
        return list


    def __len__(self):
        return len(self.dict)

import os
import pickle
from Pword2sequence import Word2Squence


import dataset

if __name__ == '__main__':

    ws=Word2Squence()
    train_data_path = r'D:\Project\Python\Data\aclImdb\train'
    list_path=[os.path.join(train_data_path,"pos"),os.path.join(train_data_path,"neg")]
    for path in list_path:
        file_name_list=os.listdir(path) #文件名字
        path_list_temp=[os.path.join(path,file_name) for file_name in file_name_list]#绝对路径拼接
        for file_path in path_list_temp:
            sentence=dataset.tokenize(open(file_path,encoding='mac_roman').read())
            ws.fit(sentence)

    ws.build_vocabulary(min=10)

    pickle.dump(ws,open('./model/ws.pkl','wb'))
    print(len(ws))

#
# if __name__ == '__main__':
#     ws=Word2Squence()
#     ws.fit(["我","爱","你"])
#     ws.fit(["我", "爱", "中国"])
#     ws.build_vocabulary() #对dict和inverse_dict两个字典进行重构
#     print(ws.count) #{'我': 2, '爱': 2, '你': 1, '中国': 1}
#     print(ws.dict) #{'UNK': 0, 'PAD': 1, '我': 2, '爱': 3, '你': 4, '中国': 5}
#     sentence=['我',"爱","和平"]
#     index=ws.transform(sentence,6)
#     print(index)#[2, 3, 0, 1, 1, 1]
#     print(ws.inverse_transform(index))#['我', '爱', 'UNK', 'PAD', 'PAD', 'PAD']
#








