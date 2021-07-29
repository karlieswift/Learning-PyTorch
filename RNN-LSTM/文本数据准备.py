"""
 Env: /anaconda3/python3.7
 Time: 2021/7/23 10:39
 Author: karlieswfit
 File: 文本数据准备.py
 Describe: 数据网站 https://ai.stanford.edu/~amaas/data/sentiment/
 数据准备 模型构建 训练 评估
"""
import torch
from torch.utils.data import DataLoader,Dataset
import os
import re

#1. 定义tokenize的方法
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
        ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
    text = re.sub("<.*?>"," ",text,flags=re.S) #去除<br />
    text = re.sub("|".join(fileters)," ",text,flags=re.S)#将特殊字符用"|"或运算连接 去除特殊字符
    return [i.strip().lower() for i in text.split()]



class ImdbDataSet(Dataset):
    def __init__(self,train=True):
        self.train_data_path=r'D:\Project\Python\Data\aclImdb\train'
        self.test_data_path=r'D:\Project\Python\Data\aclImdb\test'
        data_path=self.train_data_path if train else self.test_data_path
        # todo 1-把所有的文件名的绝对路径放入list
        self.all_data_path=[] #所有文件的绝对路径

        # 读取文件夹下的两个目录pos neg 进行路径拼接
        temp_data_path=[os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
        for path in temp_data_path:
            #获得文件夹下的文件名 在和路径拼接
            file_name_list=os.listdir(path) #返回文件列表
            file_path_list=[os.path.join(path,name) for name in file_name_list]
            self.all_data_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path=self.all_data_path[index]
        #获取label
        label_str=file_path.split("\\")[-2]
        label= 0 if label_str=="neg" else 1
        #获取内容
        content=open(file_path).read()
        #对文本数据进行提取纯单词
        tokens=tokenize(content)
        return tokens,int(label)

    def __len__(self):
        return len(self.all_data_path)



def collate_fn(batch):
	#batch是list返回batch_size大小的__getitem__
    content,label = list(zip(*batch))
    return content,label

def get_dataloader(train=True):
    imdb_data=ImdbDataSet(train)
    # data=DataLoader(imdb_data,batch_size=2,shuffle=True)   #会报错  RuntimeError: each element in list of batch should be of equal size
    data=DataLoader(imdb_data,batch_size=4,shuffle=True,collate_fn=collate_fn)
    '''
    data的形式是(index,input,target)
    index=大小总样本/batch_size 从0开始
    input=（batch_size大小的list 每个list 由单词组成 是一个句子）
    target=(是一个标签，大小为batch_size)
    '''
    return  data



if __name__ == '__main__':

    for index,(input,target) in enumerate(get_dataloader()):
        print(index)
        print(input)
        print(target)
        break
































