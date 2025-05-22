import time
import numpy as np
from application.myload_encf_todb import get_tFeature
import tenseal as ts
user_id='20240507161522pjWorwc17UAXSXb'
'''f1=get_tFeature('20240507161522pjWorwc17UAXSX.old.png',0, mask=None, binary=False)
f2=get_tFeature('20240507161522pjWorwc17UAXSX.new.png',0, mask=None, binary=False)'''
f3=np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.new.txt' % user_id)
f4=np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.old.txt' % user_id)
def cal_dist(enc1,enc2):#计算密文的平方欧氏距离
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    return euclidean_squared  
#参数设置
for (poly_mod, coeff_mod_bit_sizes, prec) in [
            (16384,[60,40,40,40,40,40,40,60],40),
            (8192, [60, 40, 40, 60], 40),
            (4096, [40, 20, 40], 20),
            (2048, [18, 18, 18], 16),
        ]:
    context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_mod, coeff_mod_bit_sizes=coeff_mod_bit_sizes)   
    context.global_scale = pow(2, prec)
    context.generate_galois_keys()
    enc2=ts.ckks_vector(context,f4)
    #加密
    start_encrypt = time.time()
    enc1=ts.ckks_vector(context,f3)
    end_encrypt = time.time()
    encrypt_time = end_encrypt - start_encrypt
 
    #解密
    start_decrypt = time.time()
    f=enc2.decrypt()
    end_decrypt = time.time()
    decrypt_time = end_decrypt - start_decrypt

    #比对
    start_compare = time.time()
    dist=cal_dist(enc1,enc2)
    end_compare = time.time()
    compare_time = end_compare - start_compare
    print('加密用时：',encrypt_time,';解密用时：',decrypt_time,';特征密文比对计算用时：',compare_time)

'''context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[40,30,30,30,30,40])   
context.global_scale = pow(2, 30)
context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[30,25,25,25,25,25,25,30])   
context.global_scale = pow(2, 25)
context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[24,20,20,20,24])   
context.global_scale = pow(2, 20)
context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40,40,40,40,40, 60])   
context.global_scale = pow(2, 40)
context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 50, 50, 50,50,50,50,60])   
context.global_scale = pow(2, 50)
context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])   
context.global_scale = pow(2, 40)'''