"""第2章 利用用户行为数据
基于领域的算法 -> 基于用户的协同过滤算法
"""
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


train = {
    'A': {'a', 'b', 'c'},
    'B': {'a', 'c'},
    'C': {'b', 'c'},
    'D': {'c', 'd', 'e'}
}


# 读入评分数据
ratings = pd.read_table(
    './input/ml-1m/ratings.dat',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    engine='python',  # 默认引擎是'c'，但是'c'引擎不支持大于一个字符的分隔符
    sep='::'
    )

# 构造训练集和测试集
train, test = train_test_split(ratings, test_size=0.125, random_state=42)


# 评测函数


# 定义相似度计算函数

def UserSimilarity(train):
    '''遍历整个数据集，计算用户相似度'''
    user_items = ratings.groupby('UserID')['MovieID'].agg(lambda x: set(x))
    W = dict()
    for u in user_items.index:
        W[u] = dict()
        for v in user_items.index:
            if u == v:
                continue
            W[u][v] = len(user_items[u] & user_items[v])
            W[u][v] /= np.sqrt(len(user_items[u]) * len(user_items[v]))
    return W


def UserSimilarity2(train):
    '''计算用户相似度
      先遍历倒排表，得到有交叉的用户数据，然后再计算相似度，避免计算很多无交集的用户组合
      感觉可以再省掉一半的计算量
    '''
    train = train.sample(10000)
    item_users = train.groupby('MovieID')['UserID'].agg(lambda x: set(x))
    C = dict()
    N = dict()
    for item, users in item_users.iteritems():
        for u in users:
            N.setdefault(u, 0)
            N[u] += 1
            if u not in C:
                C[u] = dict()
            for v in users:
                if u == v:
                    continue
                C[u].setdefault(v, 0)
                C[u][v] += 1
    
    W = dict()
    for u, related_users in C.items():
        if u not in W:
            W[u] = dict()
        for v, cnt in related_users.items():
            if u == v:
                continue
            W[u][v] = cnt / np.sqrt(N[u] * N[v])
    return W



