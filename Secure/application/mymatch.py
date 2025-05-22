import tenseal as ts
import sqlite3
import numpy as np
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from application.models import User, db
import os
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent
def cal_dist(enc1,enc2):#计算密文的平方欧氏距离
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    return euclidean_squared
def cal_all_similarity_from_db(feat):
    min_distance = float('inf')#初始化最小距离
    matching_name = None#匹配的名字
    query_result = User.query.all()
    ctx=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))
    enc_feat=ts.ckks_vector(ctx,feat)
    for row in query_result:
        name = row.user_id
        encrypted_feature_vector_db = row.enc_feature
        enc2=ts.lazy_ckks_vector_from(encrypted_feature_vector_db)
        enc2.link_context(ctx)
        # 计算欧式距离
        distance = cal_dist(enc_feat, enc2)
        distance=distance.decrypt()[0]
        print('distance',distance)
        # 更新最小距离和对应的名称
        if distance <= min_distance:
            min_distance = distance
            matching_name = name
    
    min_dist=abs(min_distance) 
    if min_dist<=0.64:
        # 输出结果
        print("匹配的名称：", matching_name)
        print('平方欧式距离：',min_dist)
        return matching_name,min_dist
    else:
        return None,None
   
def cal_all_similarity(feat):
    min_distance = float('inf')#初始化最小距离
    matching_name = None#匹配的名字
    ctx=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))
    enc_feat=ts.ckks_vector(ctx,feat)
    folder_path = './faceset/facedata'
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是以 .txt 结尾的文本文件
        if filename.endswith('.txt'):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, filename)
            enc3_proto=read_data(file_path)
            enc3=ts.lazy_ckks_vector_from(enc3_proto)
            enc3.link_context(ctx)
                # 计算欧式距离
            distance = cal_dist(enc_feat, enc3)
            distance=distance.decrypt()[0]
            distance=abs(distance)
            print('distance',distance)
                # 更新最小距离和对应的名称
            if  distance<= min_distance:
                min_distance = distance
                matching_name = os.path.splitext(filename)[0]            
    if min_distance<=0.64:
                # 输出结果
        print("匹配的名称：", matching_name)
        print('平方欧式距离：',min_distance)
        return matching_name,min_distance
    else:
        return 'None','None'