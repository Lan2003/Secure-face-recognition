# -*- coding: utf-8 -*-#得到特征值，把库里的图片获取密文存入指定文件夹
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

if __name__=='__main__':
    context=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))
    img_folder = 'D:/Face/Secure/faceset'
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'): 
            username = os.path.splitext(filename)[0]       
            feature = get_tFeature1(filename,0, mask=None, binary=False)   
            print(username,feature)          
            enc_feature=ts.ckks_vector(context,feature) 
            enc=enc_feature.serialize()
            write_data('D:/Face/Secure/faceset/facedata/'+username+'.txt',file_content=enc)