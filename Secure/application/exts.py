# -*- coding: utf-8 -*-

import os
import gzip
import base64
import numpy as np


# 图片后缀类型判断
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']


# 获取原始数据
def get_origin_data(user_id,param):
    path = './application/data/originUserFaceData/'
    if os.path.exists(path + "%s.%s.txt" % (user_id,param)):
        face_data_a = np.loadtxt(path + user_id + '.%s.txt' % param , delimiter=",")
        face_data_b = face_data_a.tolist()
        return face_data_b
    else:
        return None


# 获取加密数据
def get_encrypt_data(user_id,param):
    path = './application/data/encryptUserFaceData/'
    print(user_id,param)
    if os.path.exists(path + "%s.%s.txt" % (user_id,param)):
        with open(path + "%s.%s.txt" % (user_id,param), "rb") as f:
            encrypt_res = f.read()
        encrypt_res = str(base64.b64encode(gzip.compress(encrypt_res)))
        #print('encrypt_res: ',encrypt_res)
        return encrypt_res
    else:
        return None
if __name__== '__main__':
    user_id='3'
    #print(get_origin_data(user_id,'old'))
    #print(get_origin_data(user_id,'new'))
    print(get_encrypt_data(user_id,'old'))
    print(get_encrypt_data(user_id,'new'))