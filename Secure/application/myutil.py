# -*- coding: utf-8 -*-
import tenseal as ts
import os
import numpy as np
import socket
import time
from application.myload_encf_todb import get_tFeature
from application.seal import seals #改 删application.,单个运行，总体要加回来
from application.seal.myseal import encrypt,decrypt
ctx_path='./application/data/ctx/user/'
def write_data(filename,file_content):
    with open(filename,'wb') as f:
        f.write(file_content)
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent      
def cal_dist(enc1,enc2):#计算密文的平方欧氏距离
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    return euclidean_squared  
def gen_user_key(user_id):
    
    context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40,28,40])   
    context.global_scale = pow(2, 28)
    #context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60])   
    #context.global_scale = pow(2, 40)
    context.generate_galois_keys()  # 生成伽罗瓦密钥  
    secret_context=context.serialize(save_secret_key=True)
    write_data(ctx_path+'%s_secret.txt' %user_id,file_content=secret_context)
    context.make_context_public()
    public_context=context.serialize()
    write_data(filename=ctx_path+'%s_public.txt' %user_id,file_content=public_context)
    if os.path.exists(ctx_path+'%s_public.txt' %user_id):
        return True
    return False

# type:new or old 加密新老照片人脸数据
def data_encrypt(user_id,type):
    encrypt_path = './application/data/encryptUserFaceData/' + user_id + '.' + type+'.txt'
    face_data_path = './application/data/originUserFaceData/'

    face_data = []
    ctx=ts.context_from(read_data(ctx_path+'%s_public.txt'%user_id))
    
    if os.path.exists(face_data_path + user_id + ".%s.txt" % type):
        face_data = np.loadtxt(face_data_path + user_id + ".%s.txt" % type, delimiter=",")
        
    enc=ts.ckks_vector(ctx,face_data)
    enc=enc.serialize()

    with open(encrypt_path, "wb") as f:
        f.write(enc)
    if os.path.exists(encrypt_path):
        return True
    return False
#注册时，将加密特征放在daceset/facedata里
def data_enc_save(user_id):
    save_path = './faceset/facedata/' + user_id + '.txt'
    ctx=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))
    face_data=get_tFeature(user_id+'.old.png',0, mask=None, binary=False) 
    enc=ts.ckks_vector(ctx,face_data)
    ence=enc.serialize()

    with open(save_path, "wb") as f:
        f.write(ence)
    return ence

#本地计算
def face_compares(user_id):
    encrypt_path = './application/data/encryptUserFaceData/' + user_id
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))
    
    enc1_proto=read_data(encrypt_path + ".old.txt")
    enc1=ts.lazy_ckks_vector_from(enc1_proto)
    enc1.link_context(ctx)
    enc2_proto=read_data(encrypt_path + ".new.txt")
    enc2=ts.lazy_ckks_vector_from(enc2_proto)
    enc2.link_context(ctx)
    dist=cal_dist(enc1, enc2)
    result = dist.decrypt()[0]
    return result
#发送给服务器计算
def face_com3(user_id):
    encrypt_path = './application/data/encryptUserFaceData/' + user_id
    ct_p=ctx_path+'%s_public.txt'%user_id
    c1_p=encrypt_path + ".old.txt"
    c2_p=encrypt_path + ".new.txt"
    
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(('localhost',8888))
    #1
    fsize=os.path.getsize(ct_p)
    client.send(str(fsize).encode('utf-8'))
    ctx=read_data(ct_p)
    client.sendall(ctx)
    print('context发送完毕')
    #等待服务器接收文件完毕
    confirmation=client.recv(1024).decode()
    print(confirmation)
    #2
    fsize=os.path.getsize(c1_p)
    client.send(str(fsize).encode('utf-8'))
    c1=read_data(c1_p)
    client.sendall(c1)
    print('密文特征c1发送完毕')
    #等待服务器接收文件完毕
    confirmation=client.recv(1024).decode()
    print(confirmation)
    #3
    fsize=os.path.getsize(c2_p)
    client.send(str(fsize).encode('utf-8'))
    c2=read_data(c2_p)
    client.sendall(c2)
    print('密文特征c2发送完毕')
    confirmation=client.recv(1024).decode()
    print(confirmation)
    #接收结果
    resize=int(client.recv(1024).decode('utf-8'))
    file_data=b''
    while len(file_data)<resize:
        remain=resize-len(file_data)
        chunk= client.recv(min(remain,1024))
        if not chunk:
            break
        file_data+=chunk
    write_data(ctx_path+'%s_result.txt'%user_id,file_data)
    context=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))
    result=read_data(ctx_path+'%s_result.txt'%user_id)
    resu=ts.lazy_ckks_vector_from(result)
    resu.link_context(context)
    resu=resu.decrypt()[0]
    return resu
    
    #client.close()
    
    
    
if __name__ == '__main__':
    # 所有path路径是相对于main.py而言的，如需单独运行，则需要去除每个path中的/application
    #gen_user_key('20240315192430bGr99xHhqTw8ky2')
    user_id='202404171525532nv7mASziF84D5i'
    #gen_user_key(user_id)
    data_encrypt(user_id, 'old')
    data_encrypt(user_id,'new')
    dist=face_compares(user_id)
    print(dist)
    dist=face_com3(user_id)
    print(dist)
