"""第2章 利用用户行为数据
基于领域的算法 -> 基于物品的协同过滤算法
"""
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from operator import itemgetter
from sklearn.model_selection import train_test_split


train = [
    ('A', 'a'),
    ('A', 'b'),
    ('A', 'd'),
    ('B', 'a'),
    ('B', 'c'),
    ('C', 'b'),
    ('C', 'e'),
    ('D', 'c'),
    ('D', 'd'),
    ('D', 'e')
]

train = pd.DataFrame(train, columns=['UserID', 'MovieID'])

# 读入评分数据
ratings = pd.read_table(
    './input/ml-1m/ratings.dat',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    engine='python',  # 默认引擎是'c'，但是'c'引擎不支持大于一个字符的分隔符
    sep='::'
    )

# 构造训练集和测试集
# 因为点评行为有时间顺序，所以将早期的数据作为训练集，后期的数据作为测试集
def split_dataset(dataset, test_size=0.2):
    train_idx = np.ceil(dataset.shape[0] * (1 - test_size))
    return dataset[:train_idx], dataset[train_idx:]

train, test = split_dataset(ratings)

# 评测函数


# 定义相似度计算函数
def ItemSimilarity(train):
    user_items = train.groupby('UserID')['MovieID'].agg(lambda x: set(x))
    C = dict()  # 共现矩阵
    N = dict()  # 喜欢每个物品的人数
    for u in user_items.index:
        for i in user_items[u]:
            N[i] += 1
            C[i] = dict()
            for j in user_items[u]:
                if i == j:
                    continue
                C[i][j] += 1
     W = dict()
     for i in C.keys():
        W[i] = dict()
        for j in C[i].keys():
            W[i][j] = C[i][j] / np.sqrt(N[i] * N[j])  # 用几何平均数避免普通物品和
                                                      # 热门物品相似的问题
    return W


def recommendation(train, user_id, W, k):
    rank = dict()
    ru = train[user_id]