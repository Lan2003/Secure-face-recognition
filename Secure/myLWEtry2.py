import numpy as np
import os
import random
import time
import math
import h5py
from application.myload_encf_todb import get_tFeature1,get_tFeature#application.
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
def isPrime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

#生成参数
def genParameter(d, cla):#d:维度 cla:类别数
    MO = np.random.rand( d, d)#d x d 的矩阵MO
    MU = np.linalg.inv(MO)#MO的逆矩阵
    p1 = random.getrandbits(32)  # 生成32比特的随机整数
    p2 = random.getrandbits(8)
    while (1):
        if not isPrime(p1):
            p1 += -1
        else:
            break
    while (1):
        if not isPrime(p2):
            p2 += -1
        else:
            break
    
    while (1):#生成随机数gamma，确保大于2 * abs(np.max(xi1)
        Gamma = random.randint(1, p1)#缩放因子
        xi1 = np.random.randint(low=1, high=p2, size= d)#长为d 的随机数组
        if Gamma > 2 * abs(np.max(xi1)):
            break
    keyF = [MO, Gamma, p2,xi1]
    keyQ = [MU, Gamma, p1, p2,xi1]
    #print('keyF:',keyF)
    return keyF,keyQ

def encFeat(feat, keyF):#对单个特征feat加密，使用密钥keyF
    MO, Gamma,  p2,xi = keyF
    EncFeat = np.dot((Gamma * feat + xi), MO)
    return EncFeat# 1 x 2len(feat)的向量

def encQuery(query, keyQ):#对单个查询向量加密，用keyQ加密
    MU, Gamma, p1, p2,xi = keyQ
    Size = len(query)  # 查询向量的维数
    EncQuery = np.dot(MU, (Gamma * query.T + xi.T))  # 加密查询向量
    return EncQuery

# 欧氏距离
def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


# 密文欧氏距离计算
def calculate(encFeat, encQuery, keyQ):
    MU, Gamma, p1, p2,xi = keyQ
    dis = (np.dot(encFeat, encQuery) / (Gamma ** 2)) % p1
    return dis

# 明文欧氏距离计算
def plainCalculate(feat, query):
    dis = np.sqrt(sum(np.power((feat - query), 2)))
    return dis

if __name__=='__main__':
    user_id='20240430095714b791sfgTsS7p8UC'
    KF,KQ=genParameter(512,1)

    #face_data1 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.old.txt' % user_id)
    face_data1=get_tFeature('20240430095714b791sfgTsS7p8UC.old.png',0,None,False)
    face_data2 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.new.txt' % user_id)

    dism=plainCalculate(face_data1,face_data2)
    print('明文距离dism:',dism) 
    f=encQuery(face_data1,KQ)
    q=encFeat(face_data2,KF)
    disc=calculate(f,q,KQ)  
    print('密文距离disc:',disc)

    
    img_folder = 'D:/Face/Secure/faceset'
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'):        
            feature = get_tFeature1(filename,0, mask=None, binary=False)   
            #print(feature) 
            dism=plainCalculate(face_data1,feature)         
            enc_feature=encFeat(feature,KF)  
            disc=calculate(f,enc_feature,KQ)
            
            print('明文距离：',dism,'密文距离:',disc)
'''import numpy as np
import os
import random
from application.face_datamy import get_Feature1
np.set_printoptions(threshold=np.inf)
import time
import math
def isPrime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

#生成参数
def genParameter(d, cla):#d:维度 cla:类别数
    MO = np.random.rand(2 * d, 2 * d)#2d x 2d 的矩阵MO
    MU = np.linalg.inv(MO)#MO的逆矩阵
    p1 = random.getrandbits(32)  # 生成32比特的随机整数
    p2 = random.getrandbits(8)
    while (1):
        if not isPrime(p1):
            p1 += -1
        else:
            break
    while (1):
        if not isPrime(p2):
            p2 += -1
        else:
            break
    Alpha = np.random.randint(low=1, high=p2, size=d - 1)#长为d-1的数组，元素为1-p2之间的整数
    while (1):#生成随机数gamma，确保大于2 * abs(np.max(xi1)
        Gamma = random.randint(1, p1)#缩放因子
        xi1 = np.random.randint(low=1, high=p2, size=2 * d)#长为2d 的随机数组
        if Gamma > 2 * abs(np.max(xi1)):
            break
    keyF = [MO, Gamma, Alpha, p2]
    keyQ = [MU, Gamma, p1, p2]
    #print('keyF:',keyF)
    return keyF,keyQ

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

# 密文欧氏距离计算
def calculate(encFeat, encQuery, keyQ):
    MU, Gamma, p1, p2 = keyQ
    print('p1:',p1,'Gammma:',Gamma)
    dis = (np.dot(encFeat, encQuery) / (Gamma ** 2)) % p1
    return dis


# 明文欧氏距离计算
def plainCalculate(feat, query):
    dis = np.sqrt(sum(np.power((feat - query), 2)))
    return dis

if __name__=='__main__':
    user_id='20240430095714b791sfgTsS7p8UC'
    KF,KQ=genParameter(512,1)
    #print(KF)
    face_data1 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.old.txt' % user_id)
    face_data2 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.new.txt' % user_id)
    face_data3 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/20240429195320sAr50fE6x2iUpt2.old.txt' )
    #print(face_data1)
    dism=plainCalculate(face_data1,face_data2)
    print('明文距离dism:',dism)
    
    f=encFeat(face_data1,KF)
    q=encQuery(face_data2,KQ)
    q1=encQuery(face_data3,KQ)
    #print(f)
    disc=calculate(f,q,KQ)
    
    print('密文距离disc:',disc)
    
    disc1=calculate(f,q1,KQ)
    
    print('密文距离disc1:',disc1)
    
    img_folder = 'D:/Face/Secure/faceset'
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'):        
            feature = get_Feature1(filename,0, mask=None, binary=False)   
            print(feature)          
            enc_feature=encQuery(feature,KQ)  
            disc=calculate(f,enc_feature,KQ)
            
            print('disc:',disc)'''
    
    