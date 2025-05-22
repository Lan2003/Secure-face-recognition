import numpy as np
import h5py
import random

np.set_printoptions(threshold=np.inf)


def saveFeatures(dir, Feats, Paths):#将特征Feats和对应路径Paths保存到指定文件dir中
    h5y = h5py.File(dir, 'w')
    h5y.create_dataset('feats', data=Feats)
    h5y.create_dataset('paths', data=Paths)
    h5y.close()


def loadFeatures(dir):#加载特征和路径信息
    f = h5py.File(dir, 'r')
    feats = f['feats'][:]
    paths = f["paths"][:]
    f.close()
    return feats, paths


def encFeat(feat, keyF):#对单个特征feat加密，使用密钥keyF
    MO, Gamma, Alpha, p2 = keyF
    exFeat = np.concatenate((feat, [-0.5 * (np.linalg.norm(feat) ** 2)], Alpha), axis=0)#拼接多个数组，第二个是向量长度平方的-0.5
    while (1):
        xi = np.random.randint(low=1, high=p2, size=len(feat) * 2)#用于混淆加密过程
        if Gamma > 2 * abs(np.max(xi)):
            break
    EncFeat = np.dot((Gamma * exFeat + xi), MO)
    return EncFeat# 1 x 2len(feat)的向量

def encQuery(query, keyQ):#对单个查询向量加密，用keyQ加密
    MU, Gamma, p1, p2 = keyQ
    Size = len(query)  # 查询向量的维数
    Delta = random.randint(1, p2)
    Beta = np.random.randint(low=1, high=p2, size=Size - 1)

    while (1):
        xi = np.random.randint(low=1, high=p2, size=Size * 2)
        if Gamma > 2 * abs(np.max(xi)):
            break

    exQuery = np.concatenate((query * Delta, [Delta], Beta))
    EncQuery = np.dot(MU, (Gamma * exQuery.T + xi.T))  # 加密查询向量
    return EncQuery

def encFeats(feats, keyF):#对多个特征向量Feats批量加密，使用keyF加密
    MO, Gamma, p2 = keyF
    Alpha = np.random.randint(low=1, high=p2, size=len(feats[0]) - 1)
    a2 = -0.5 * (np.linalg.norm(feats, axis=1) ** 2).reshape((-1, 1))
    a3 = np.tile(Alpha, [len(feats), 1])
    exFeats = np.concatenate((feats, a2, a3), axis=1)
    EncFeats = np.dot((Gamma * exFeats), MO)
    return EncFeats

def encQuerys(querys, keyQ):#对多个查询向量批量加密，用keyQ加密
    MU, Deta, Belta = keyQ
    a2 = np.random.randint(1, 2, size=(len(querys), 1))
    a3 = np.tile(Belta, [len(querys), 1])
    exQuerys = Deta * np.concatenate((querys, a2, a3), axis=1)
    EncQuerys = np.dot(exQuerys, MU)
    return EncQuerys
