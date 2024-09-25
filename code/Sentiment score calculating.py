# -*- coding: utf-8 -*-

import jieba
import re
from snownlp import SnowNLP
import pandas as pd
import numpy as np

from operator import add
from functools import reduce


X_train = pd.read_csv("main\data\Social Reviews for Jiuzhaigou (text).csv");
X_train = np.array(X_train)
nn=len(X_train)
n=0

sent=list(range(1))
sent=pd.DataFrame(sent)
myset1=list(range(nn))

while n <nn:
    data1 = X_train[n,1]
    data2 = pd.DataFrame([sentence for sentence in str(data1).split('.') if sentence],columns=['sentences'])
    data2.to_csv('main\data\data2.csv',encoding='utf_8_sig')
    with open('main\data\data2.csv', encoding='utf-8') as f:
        data = f.read()
 
# Text preprocessing: removes useless characters and extracts only Chinese characters.
    new_data = re.findall('[\u4e00-\u9fa5]+', data, re.S)
    new_data = "/".join(new_data)
 
# Text Segmentation
    seg_list_exact = jieba.cut(new_data, cut_all=False)
 
# Load stop-word data
    with open("main\code\Chinese stopword list.txt", encoding='utf-8') as f:
    # Get the stop words for each line and add them to the collection.
        con = f.read().split('\n')
        stop_words = set()
    for i in con:
        stop_words.add(i)
 
# Removal of stop words and single words
    result_list = [word for word in seg_list_exact if word not in stop_words and len(word) > 1]
    u=len(result_list)
    myset=result_list[0]+'/';
    j=1;
    while j <u:
        new_list=result_list[j]+'/'
        myset=reduce(add,(myset,new_list))
        j=j+1
    myset1[n]=myset;
    u=len(result_list)
    ss=list(range(u))    
    text=myset;
    s = SnowNLP(text)
    ss = s.sentiments
    sent[n]=ss;
    n=n+1
sent=sent.T
sent.to_excel("Social Sentiment score for Jiuzhaigou.xlsx",index=False)