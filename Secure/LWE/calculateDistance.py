import numpy as np


# 欧氏距离
def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


# 密文欧氏距离计算
def calculate(encFeat, encQuery, keyQ):
    MU, Gamma, p1, p2 = keyQ
    dis = (np.dot(encFeat, encQuery) / (Gamma ** 2)) % p1
    return dis


# 明文欧氏距离计算
def plainCalculate(feat, query):
    dis = np.sqrt(sum(np.power((feat - query), 2)))
    return dis
