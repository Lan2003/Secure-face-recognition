import random
import os
import math
import numpy as np
from application.face_datamy import get_Feature1
from application.util import face_compares, gen_user_key, data_encrypt
from application.myload_encf_todb import get_tFeature1,get_tFeature#application.
n_lwe = 512#3000
s = 8
p = 2 ** 48 + 1
q = 2 ** 77
l = 512

class PublicKey:
    def __init__(self, A, P, n_lwe, s):
        self.A = A
        self.P = P
        self.n_lwe = n_lwe
        self.s = s

    def __repr__(self):
        return 'PublicKey({}, {}, {}, {})'.format(self.A, self.P, self.n_lwe, self.s)

class Ciphertext:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        return 'Ciphertext({}, {})'.format(self.c1, self.c2)

    def __add__(self, c):
        c1 = []
        for i in range(n_lwe):
            c1.append(self.c1[i] + c.c1[i])

        c2 = []
        for i in range(l):
            c2.append(self.c2[i] + c.c2[i])

        return Ciphertext(c1, c2)
    def __sub__(self, c):
        c1 = []
        for i in range(n_lwe):
            c1.append(self.c1[i] - c.c1[i])

        c2 = []
        for i in range(l):
            c2.append(self.c2[i] - c.c2[i])

        return Ciphertext(c1, c2)
    def __mul__(self, c):
        c1 = []
        for i in range(n_lwe):
            c1.append(self.c1[i] * c.c1[i])

        c2 = []
        for i in range(l):
            c2.append(self.c2[i] * c.c2[i])

        return Ciphertext(c1, c2)
    def dot(self, other):
        if len(self.c1) != len(other.c1) or len(self.c2) != len(other.c2):
            raise ValueError("Length of vectors do not match")
        
        # 对 c1 分量进行点积
        dot_product_c1 = sum(x * y for x, y in zip(self.c1, other.c1))
        # 对 c2 分量进行点积
        dot_product_c2 = sum(x * y for x, y in zip(self.c2, other.c2))
        # 将两个分量的点积结果相加
        result = (dot_product_c2)%p
        return result
    def euclidean_distance(self, c):
        """
        计算两个密文之间的欧几里得距离。
        """
        if len(self.c1) != len(c.c1) or len(self.c2) != len(c.c2):
            raise ValueError("密文分量长度不一致，无法计算距离。")

        sum_of_squares = 0
        '''# 计算 c1 部分的距离
        for i in range(n_lwe):
            sum_of_squares += (self.c1[i] - c.c1[i]) ** 2'''
        
        # 计算 c2 部分的距离
        for i in range(l):
            sum_of_squares += (self.c2[i] - c.c2[i]) ** 2
        sum_of_squares=sum_of_squares%p
        return math.sqrt(sum_of_squares)

def get_discrete_gaussian_random_matrix(m, n):
    sample = []
    for i in range(m):
        row_sample = []
        for i in range(n):
            row_sample.append(round(random.gauss(0, s)))
        sample.append(row_sample)
    return sample

def get_discrete_gaussian_random_vector(n):
    sample = []
    for i in range(n):
        sample.append(round(random.gauss(0, s)))
    return sample

def get_uniform_random_matrix(m, n):
    sample = []
    for i in range(m):
        row_sample = []
        for i in range(n):
            row_sample.append(random.randint(-q // 2 + 1, q // 2))
        sample.append(row_sample)
    return sample

def KeyGen():
    R = get_discrete_gaussian_random_matrix(n_lwe, l)
    S = get_discrete_gaussian_random_matrix(n_lwe, l)
    A = get_uniform_random_matrix(n_lwe, n_lwe)

    P = []
    for i in range(n_lwe):
        row_P = []
        for j in range(l):
            value = p * R[i][j]
            for tmp in range(n_lwe):
                value -= A[i][tmp] * S[tmp][j]
            row_P.append(value % p)
        P.append(row_P)
    return PublicKey(A, P, n_lwe, s), S
def KeyGen1():
    e = get_discrete_gaussian_random_matrix(n_lwe, l)
    S = get_discrete_gaussian_random_matrix(n_lwe, l)
    A = get_uniform_random_matrix(n_lwe, n_lwe)

    b = []
    for i in range(n_lwe):
        row_P = []
        for j in range(l):
            value = e[i][j]
            for tmp in range(n_lwe):
                value += A[i][tmp] * S[tmp][j]
            row_P.append(value % p)
        b.append(row_P)
    return PublicKey(A, b, n_lwe, s), S
def Enc(pk, m):
    e1 = get_discrete_gaussian_random_vector(n_lwe)
    e2 = get_discrete_gaussian_random_vector(n_lwe)
    e3 = get_discrete_gaussian_random_vector(l)

    c1 = []#At*e2
    for i in range(n_lwe):
        value = p * e2[i]
        for tmp in range(n_lwe):
            value += e1[tmp] * pk.A[tmp][i]
        c1.append(value)

    c2 = []#Pt*e3+enc(m) (P=A*s'+)
    for i in range(l):
        value = p * e3[i] + m[i]#初步简单加密明文 enc(m)
        for tmp in range(n_lwe):#混淆加密
            value += e1[tmp] * pk.P[tmp][i]
        c2.append(value)
    return Ciphertext(c1, c2)

def Dec(S, c):
    m = []
    for i in range(l):
        value = c.c2[i]
        for tmp in range(n_lwe):
            value -= c.c1[tmp] * S[tmp][i]#原来是+
        m.append(value % p)
    return m

def Oushi(c1,c2):
    c=c1-c2
    res=c.dot(c)
    return res
def cosine_similarity(c1, c2):
    numerator = c1.dot(c2) 
    numerator=numerator*numerator
    denominator =c1.dot(c1)*c2.dot(c2)
    if denominator == 0:
        return 0  # Handle division by zero
    return numerator / denominator
def simil2(f1, f2):#cosine_similarity
    # compute cosine_similarity for 2-D array
    #f1 = f1.numpy()
    #f2 = f2.numpy()

    A = np.sum(f1*f2, axis=0)
    B = np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0) + 1e-5

    return A / B  
user_id='20240430095714b791sfgTsS7p8UC'
pk, sk = KeyGen()
face_data_a = np.loadtxt('./application/data/originUserFaceData/' + user_id + '.old.txt'  , delimiter=",")
face_data_b = np.loadtxt('./application/data/originUserFaceData/' + user_id + '.new.txt'  , delimiter=",")
c1=Enc(pk,face_data_a)
c2=Enc(pk,face_data_b)
#m=Dec(sk,c1)

#print('m:',m)
#print('c1:',c1)
#print('c2:',c2)
def calculate(c1,c2):
    
    c3=c1-c2
    dimi=c3.dot(c3)
    return dimi
#mcos=cosine_similarity(c1,c2)
#print('mcos',mcos)
'''with open('./application/datamy/originData/'+user_id+'.old', "wb") as f:
    f.write(c1)
#np.savetxt('./application/datamy/originData/'+user_id + '.old', c1)
with open('./application/datamy/originData/'+user_id+'.new', "wb") as f:
    f.write(c2)'''
'''c=c1-c2
m=Dec(sk,c)
print('c:',c)
print('m:',m)'''
c_cos1=simil2(face_data_a,face_data_b)

#c_cos2=simil2(c1,c2)

#mc2=Dec(sk,c_cos2)

print('mc1:',c_cos1)
#print('mc2:',mc2)
face_data1=get_tFeature('20240430095714b791sfgTsS7p8UC.old.png',0,None,False)
face_data2 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.new.txt' % user_id)
f=Enc(pk,face_data1)
'''
dism=simil2(face_data1,face_data2)
print('明文距离dism:',dism) 

q=Enc(pk,face_data2)
disc=calculate(f,q)  
print('密文距离disc:',disc)'''

    
img_folder = 'D:/Face/Secure/faceset'
for filename in os.listdir(img_folder):
    if filename.endswith('.jpg'):        
        feature = get_tFeature1(filename,0, mask=None, binary=False)   
            #print(feature) 
        #dism=simil2(face_data1,feature)         
        enc_feature=Enc(pk,feature)  
        #disc=calculate(f,enc_feature)
        dist=f.euclidean_distance(enc_feature)
        print('密文距离：',dist)
'''    
st = time.time()
pk, sk = KeyGen()
print("KeyGen Time: %.6f s" % (time.time() - st))

m1 = []
m2 = []
for i in range(l):
    m1.append(i*i)
    m2.append(i)
print('m1:',m1)
print('m2:',m2)
st = time.time()
c1 = Enc(pk, m1)
c2 = Enc(pk, m2)
print('c1:',c1)
print('c2:',c2)
print("Encrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / (2 * l)))

st = time.time()
c = c1 + c2
print("Add Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))

st = time.time()
m = Dec(sk, c)
print("Decrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))
print('m:',m)'''