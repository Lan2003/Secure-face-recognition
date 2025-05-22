# -*- coding: utf-8 -*-#对比两种特征得到的距离的不同
import os
import time
import base64
import string
import random
import config
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from application.models import User, db
from application.exts import allowed_file,  get_encrypt_data
from application.face_data import get_user_data
from application.face_datamy import get_Feature1
from application.detect import load_and_detect_data
from application.util import face_compares, gen_user_key, data_encrypt
from flask import render_template, request, jsonify, session, redirect


def face_compare():#对遮挡模型的特征提取
    #res = {'code': 0, 'msg': '禁止'}
    #验证用户身份
    #user_id = '20240327153400r8aIaTT7lPQDBtY'
    #req_user_id = request.cookies.get('user_id')
  
    origin_res = get_Feature1(user_id,'old',0, mask=None, binary=False) 
    encrypt_res = data_encrypt(user_id, 'old')
    origin_new_res = get_Feature1(user_id,'new',0, mask=None, binary=False)
    encrypt_new_res = data_encrypt(user_id, 'new')
 
    if origin_new_res and encrypt_new_res and origin_res and encrypt_res:
        result = face_compares(user_id)#汉明距离
        print('识别结果:', end='')
        print(result[0])
        # 关键判断部分
        if result[0] > 0.7:
            print('认证失败')
        else:
            print('认证成功')
    return result
def face_compare2():#facenet模型特征
    #res = {'code': 0, 'msg': '禁止'}
    #验证用户身份
    #user_id = '20240327153400r8aIaTT7lPQDBtY'
    #req_user_id = request.cookies.get('user_id')
    
    origin_res = get_user_data(user_id,'old')
   
    encrypt_res = data_encrypt(user_id, 'old')
            #print('')
        #origin_new_res = get_user_data(user_id,'new')
    origin_new_res = get_user_data(user_id,'new')
    encrypt_new_res = data_encrypt(user_id, 'new')
    
    #判断识别结果是否大于阈值0.7，如果大于，则认证失败，返回结果
    #print('判断真假', origin_new_res.any(), encrypt_new_res.any(), origin_res.any(), encrypt_res.any())

    if origin_new_res and encrypt_new_res and origin_res and encrypt_res:
        result = face_compares(user_id)#汉明距离
        print('识别结果:', end='')
        print(result[0])
        # 关键判断部分
        if result[0] > 0.7:
            print('认证失败')
        else:
            print('认证成功')
    return result
if __name__ == '__main__':
    user_id='202404171525532nv7mASziF84D5i'#'20240327153400r8aIaTT7lPQDBtY'
    gen_user_key(user_id)
    face_compare()#my
    #face_compare2()
    
    
    
    