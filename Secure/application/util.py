# -*- coding: utf-8 -*-

import os
import numpy as np
from application.seal import seals #改 删application,单个运行，总体要加回来
#from seal import seals 

def gen_user_key(user_id):
    seal_key = seals.regist()#生成key \x00
    key_path = './application/data/userKey/'
    with open(key_path + "%s.pk" % user_id, "wb") as f:
        f.write(seal_key[0])
    with open(key_path + "%s.sk" % user_id, "wb") as f:
        f.write(seal_key[1])
    with open(key_path + "%s.re" % user_id, "wb") as f:
        f.write(seal_key[2])
    with open(key_path + "%s.ga" % user_id, "wb") as f:
        f.write(seal_key[3])
    if os.path.exists(key_path + "%s.pk" % user_id):
        return True
    return False

# type:new or old 加密新老照片人脸数据
def data_encrypt(user_id, type):
    encrypt_path = './application/data/encryptUserFaceData/' + user_id + '.' + type
    face_data_path = './application/data/originUserFaceData/'
    key_path = './application/data/userKey/'
    face_data = []

    if os.path.exists(face_data_path + user_id + ".%s.txt" % type):
        face_data = np.loadtxt(face_data_path + user_id + ".%s.txt" % type, delimiter=",")
    #print('face_data',face_data)
    with open(key_path + "%s.pk" % user_id, "rb") as f:
        public_key = f.read()
    #print('public_key',public_key)
    cipher_str = seals.encrypt(face_data, public_key)#用公钥加密人脸数据
    #print('cipher_str',cipher_str)
    with open(encrypt_path, "wb") as f:
        f.write(cipher_str)
    if os.path.exists(encrypt_path):
        return True
    return False


def face_compares(user_id):
    key_path = './application/data/userKey/'
    encrypt_path = './application/data/encryptUserFaceData/' + user_id

    with open(key_path + "%s.sk" % user_id, "rb") as f:
        secret_key = f.read()
    with open(key_path + "%s.re" % user_id, "rb") as f:
        relin_key = f.read()
    with open(key_path + "%s.ga" % user_id, "rb") as f:
        gal_key = f.read()

    with open(encrypt_path + ".old", "rb") as f:
        cipher1 = f.read()
    with open(encrypt_path + ".new", "rb") as f:
        cipher2 = f.read()

    res = seals.hamming(cipher1, cipher2, relin_key, gal_key)
    result = seals.decrypt(secret_key, res)
    return result


if __name__ == '__main__':
    # 所有path路径是相对于main.py而言的，如需单独运行，则需要去除每个path中的/application
    #gen_user_key('20240315192430bGr99xHhqTw8ky2')
    data_encrypt('20240315202506iyK2SXM9vIDiyNO', 'old')
