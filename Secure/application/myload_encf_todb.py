# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import time
import tenseal as ts
import numpy as np
from application.face_datamy import load_and_align_data
from application.fpn_model import LResNet50E_IR_Occ as LResNet50E_IR_FPN

from torchvision import transforms
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_path = './application/face/src/modelp5/model_p5.pth.tar'

db = SQLAlchemy()

class User(db.Model):
    __tablename = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_status = db.Column(db.Integer)
    user_upload = db.Column(db.Integer)
    user_id = db.Column(db.String(29), nullable=False)
    create_time = db.Column(db.DateTime)
    enc_feature = db.Column(db.BLOB)
class facedata(db.Model):
    __tablename = 'facedata'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(45))
    enc_feature = db.Column(db.BLOB)
    create_time = db.Column(db.DateTime)

model = LResNet50E_IR_FPN(num_mask=226)
#model = torch.nn.DataParallel(model)
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))  # 加载模型参数
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict,strict=False)  # 加载模型状态字典
model.eval()
preprocess = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
#本文件专用
def get_tFeature1(img_filename,is_gray, mask=None, binary=False):
    
    img_path = 'D:/Face/Secure/faceset/' + img_filename#名称+后缀
    img=load_and_align_data(img_path, 112,96, 44, 1.0)
    #with open(img_path, 'rb') as f:
        #img =  Image.open(f).convert('RGB')
    img=preprocess(img).unsqueeze(0)
    fc_mask, mask, vec, fc = model(img)
    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    fc_mask_array = fc_mask.detach().numpy()
    fc_mask_array = (2 * (fc_mask_array - fc_mask_array.min()) / (fc_mask_array.max() - fc_mask_array.min()) - 1)/10
    return fc_mask_array

def get_tFeature(img_filename,is_gray, mask=None, binary=False):
    
    img_path = 'D:/Face/Secure/upload/' + img_filename#名称+后缀
    img=load_and_align_data(img_path, 112,96, 44, 1.0)
    #with open(img_path, 'rb') as f:
        #img =  Image.open(f).convert('RGB')
    img=preprocess(img).unsqueeze(0)
    fc_mask, mask, vec, fc = model(img)
    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    fc_mask_array = fc_mask.detach().numpy()
    fc_mask_array = (2 * (fc_mask_array - fc_mask_array.min()) / (fc_mask_array.max() - fc_mask_array.min()) - 1)/10
    return fc_mask_array

def write_data(filename,file_content):
    with open(filename,'wb') as f:
        f.write(file_content)
        
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent


def load_enc_to_db():
    ctx_path='D:/Face/Secure/application/data/ctx/'#现在加了个-避免public.txt更换
    context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60])   
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()  # 生成伽罗瓦密钥  
    
    secret_context=context.serialize(save_secret_key=True)
    write_data(ctx_path+'secret.txt',file_content=secret_context)
    context.make_context_public()
    public_context=context.serialize()
    write_data(filename=ctx_path+'public.txt',file_content=public_context)
    
    # 遍历图像文件夹
    img_folder = 'D:/Face/Secure/faceset'
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'):        
            feature = get_tFeature1(filename,0, mask=None, binary=False)   
            print(feature)          
            enc_feature=ts.ckks_vector(context,feature)  
            enc_feature=enc_feature.serialize()
            username = os.path.splitext(filename)[0]# 获取用户名
            print('username:',username)
            # 将加密的特征和用户名存储到数据库
            data = facedata(
                user_id=username,    
                enc_feature=enc_feature,  
                create_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            )
            db.session.add(data)
            # 提交事务
            db.session.commit()
#注册用户数据入库
def save_to_db(user_id):
    ctx_path='D:/Face/Secure/application/data/ctx/user/'
    context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60])   
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()  # 生成伽罗瓦密钥  
    
    secret_context=context.serialize(save_secret_key=True)
    write_data(ctx_path+user_id+'_secret.txt',file_content=secret_context)
    context.make_context_public()
    public_context=context.serialize()
    write_data(filename=ctx_path+user_id+'_public.txt',file_content=public_context)
    
    imgpath=user_id+'.old.png'
    feature = get_tFeature(imgpath,0, mask=None, binary=False)   
    print(feature)          
    enc_feature=ts.ckks_vector(context,feature)  
    enc=enc_feature.serialize()
    write_data('D:/Face/Secure/faceset/facedata/'+user_id+'.txt',file_content=enc)

    # 将加密的特征和用户名存储到数据库
    user = User(
        user_status=1,
        user_upload=1,
        user_id=user_id,
        create_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        enc_feature=enc_feature.serialize()  
        )
    db.session.add(user)
    # 提交事务
    db.session.commit()