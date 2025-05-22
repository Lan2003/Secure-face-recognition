
import tenseal as ts
import numpy as np
import os
import math

ctx_path='D:/Face/Secure/application/data/ctx/'
def write_data(filename,file_content):
    with open(filename,'wb') as f:
        f.write(file_content)
        
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent

def simil(enc1,enc2,context):#密文的相似度
    numerator =enc1.dot(enc2)#分子
    numerator=numerator*numerator
    
    denominator=enc1.dot(enc1)*enc2.dot(enc2)#分母
    numerator.link_context(context)
    denominator.link_context(context)
    numerator_de=numerator.decrypt()
    denominator_de=denominator.decrypt()
    simi=np.sum(np.array(numerator_de))/np.sum(np.array(denominator_de))
    return simi
def simil1(enc1,enc2,context):#密文的相似度
    diff=enc1-enc2
    diff.link_context(context)
    diff=diff.decrypt()
    diff_squ=[x**2 for x in diff]
    diff_squ_sum=sum(diff_squ)
    
    return diff_squ_sum
def simil2(f1, f2):#cosine_similarity明文相似度
    # compute cosine_similarity for 2-D array
    #f1 = f1.numpy()
    #f2 = f2.numpy()

    A = np.sum(f1*f2, axis=0)
    B = np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0) + 1e-5

    return A / B  
def dist(enc1,enc2,context):#密文的距离
    diff=enc1-enc2
    dis_squ=diff.dot(diff)
    dis_squ.link_context(context)
   
    dis=dis_squ.decrypt()
    print(dis_squ)
    dis=np.sqrt(dis)
    return dis

def regist(user_id):
    # 创建同态加密上下文
    context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[24, 20, 20,20, 24])   
    context.global_scale = pow(2, 20)
    context.generate_galois_keys()  # 生成伽罗瓦密钥  
    secret_context=context.serialize(save_secret_key=True)
    write_data(ctx_path+'%s_secret.txt'%user_id,file_content=secret_context)
    context.make_context_public()
    public_context=context.serialize()
    write_data(filename=ctx_path+'%s_public.txt'%user_id,file_content=public_context)
    return 
def encrypt(data,user_id):
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))
    enc=ts.ckks_vector(ctx,data)
    enc=enc.serialize()
    return enc
def decrypt(secret_key, res,user_id):
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))
    res.link_context(ctx)
    resu=res.decrypt(secret_key)
    return resu
def hamming(cipher1, cipher2, relin_key, gal_key,user_id):
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))#
    ci1=ts.lazy_ckks_vector_from(cipher1)
    ci1.link_context(ctx)
    ci2=ts.lazy_ckks_vector_from(cipher2)
    ci2.link_context(ctx)
    dist=1-simil(ci1,ci2,ctx)
    return dist
def hamming2(cipher1, cipher2, relin_key, gal_key,user_id):
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))#
    ci1=ts.lazy_ckks_vector_from(cipher1)
    ci1.link_context(ctx)
    ci2=ts.lazy_ckks_vector_from(cipher2)
    ci2.link_context(ctx)
    dis=dist(ci1,ci2,ctx)
    return dis
def hamming3(cipher1, cipher2, relin_key, gal_key,user_id):
    ctx=ts.context_from(read_data(ctx_path+'%s_secret.txt'%user_id))#
    ci1=ts.lazy_ckks_vector_from(cipher1)
    ci1.link_context(ctx)
    ci2=ts.lazy_ckks_vector_from(cipher2)
    ci2.link_context(ctx)
    dis=simil1(ci1,ci2,ctx)
    return dis
def gen_user_key(user_id):
    seal_key = regist(user_id)#生成key \x00
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
    cipher_str = encrypt(face_data, public_key,user_id)#用公钥加密人脸数据
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

    res = hamming(cipher1, cipher2, relin_key, gal_key,user_id)
    result = decrypt(secret_key, res,user_id)
    return result


if __name__ == '__main__':
        # 使用示例
    
    user_id='3'
    pk,sk,relin_key, gal_key= regist(user_id)
    
    face_data1 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.old.txt' % user_id)
    face_data2 = np.loadtxt( 'D:/Face/Secure/application/data/originUserFaceData/'+'%s.new.txt' % user_id )

    dist2=1-simil2(face_data1,face_data2)#1-明文相似度
    print('1-明文相似度=dist2:',dist2)
    # 假设有两个加密的向量 cipher1 和 cipher2
    cipher1 = encrypt(face_data1, pk,user_id)
    cipher2 = encrypt(face_data2, pk,user_id)
    #print('cipher1:',cipher1)
    #print('\n')
    #print('cipher2:',cipher2)
    # 计算两个密文的汉明距离
    di=hamming3(cipher1, cipher2, relin_key, gal_key,user_id)#密文距离平方
    print('密文欧几里得距离平方dis_squ:',di)
    cidis=hamming2(cipher1, cipher2, relin_key, gal_key,user_id)#密文欧几里得距离
    print('密文欧几里得距离dist',cidis)
    cipher_ham = hamming(cipher1, cipher2, relin_key, gal_key,user_id)#1-密文相似度
    print('1-密文相似度=cipher_ham:',cipher_ham)
    
    #ham=decrypt(sk,cipher_ham,user_id)
    #print('ham:',ham)




'''import tenseal as ts
import numpy as np

def gencontext():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[22 ,21, 21, 21, 21, 21, 21, 21, 21, 21])
    context.global_scale = pow(2, 21)
    context.generate_galois_keys()
    return context

def encrypt(context, np_tensor):
    return ts.ckks_tensor(context, np_tensor)

def decrypt(enc_tensor):
    return np.array(enc_tensor.decrypt().tolist())

def bootstrap(context, tensor):
    # To refresh a tensor with exhausted depth. 
    # Here, bootstrap = enc(dec())
    tmp = decrypt(tensor)
    return encrypt(context, tmp)

if __name__ == "__main__":

    a = np.array([[1.,2.,3.,4.], [1.,2.,5.,4.]])
    print('a:',a)
    print('at:',a.T)
    context = gencontext()
    enc_a = encrypt(context, a)
    print('enc_a:',enc_a)
    enc_at = encrypt(context, a.T)
    print('enc_at:',enc_at)
    enc_b = encrypt(context, a)
    print('enc_b:',enc_b)
    res = enc_a + enc_b
    # res = enc_a - enc_b
    # res = enc_a * enc_b
    # res = enc_a @ enc_at
    print(decrypt(res))'''
'''
import tenseal as ts
import numpy as np

def gencontext():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[22 ,21, 21, 21, 21, 21, 21, 21, 21, 21])
    context.global_scale = pow(2, 21)
    context.generate_galois_keys()
    return context

def encrypt(context, np_tensor):
    return ts.ckks_tensor(context, np_tensor)

def decrypt(enc_tensor):
    return np.array(enc_tensor.decrypt().tolist())

def bootstrap(context, tensor):
    # To refresh a tensor with exhausted depth. 
    # Here, bootstrap = enc(dec())
    tmp = decrypt(tensor)
    return encrypt(context, tmp)

if __name__ == "__main__":

    a = np.array([[1.,2.,3.,4.], [1.,2.,5.,4.]])
    context = gencontext()
    enc_a = encrypt(context, a)
    enc_at = encrypt(context, a.T)
    enc_b = encrypt(context, a)
    res = enc_a + enc_b
    # res = enc_a - enc_b
    # res = enc_a * enc_b
    # res = enc_a @ enc_at
    print(decrypt(res))

# 获取图像的向量表示
def get_image_embeddings(img_path):
    return DeepFace.represent(img_path, model_name='Facenet')

# 初始化并序列化同态加密上下文
def initialize_and_serialize_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()  # 生成伽罗瓦密钥
    context.global_scale = 2**40  # 设置全局缩放因子
    secret_context = context.serialize(save_secret_key=True)  # 序列化上下文并保存密钥
    with open("secret.txt", 'wb') as f:
        f.write(secret_context)
    context.make_context_public()  # 将上下文设置为公开
    public_context = context.serialize()  # 序列化公共上下文
    with open("public.txt", 'wb') as f:
        f.write(public_context)
    del context, secret_context, public_context

# 对向量进行同态加密并写入文件
def encrypt_vector(embedding, file_name):
    context = ts.context_from("public.txt")
    enc_vector = ts.ckks_vector(context, embedding)  # 创建同态加密向量
    enc_vector_proto = enc_vector.serialize()  # 序列化加密向量
    with open(file_name, 'wb') as f:
        f.write(enc_vector_proto)
    del context, enc_vector, enc_vector_proto

# 解密并比较欧氏距离平方值
def decrypt_and_compare(euclidean_squared_file, threshold=100):
    context = ts.context_from("secret.txt")
    euclidean_squared_proto = read_data(euclidean_squared_file)  # 读取欧氏距离平方的数据
    euclidean_squared = ts.lazy_ckks_vector_from(euclidean_squared_proto)  # 使用懒惰加载从数据中恢复向量
    euclidean_squared.link_context(context)  # 将向量与上下文关联
    euclidean_squared_plain = euclidean_squared.decrypt()[0]  # 解密向量

    if euclidean_squared_plain < threshold:
        return "They are the same person"
    else:
        return "They are different persons"

# 验证结果是否一致
def validate_results(img1_embedding, img2_embedding, euclidean_squared_file):
    distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)  # 计算欧氏距离
    traditional_euclidean_squared = distance * distance

    context = ts.context_from("secret.txt")
    euclidean_squared_proto = read_data(euclidean_squared_file)
    euclidean_squared = ts.lazy_ckks_vector_from(euclidean_squared_proto)
    euclidean_squared.link_context(context)
    euclidean_squared_plain = euclidean_squared.decrypt()[0]

    if abs(traditional_euclidean_squared - euclidean_squared_plain) < 0.00001:
        return "Validation passed: Results are consistent"
    else:
        return "Validation failed: Results are not consistent"

#计算两个密文之间的汉明距离
def hamming(cipher1, cipher2, relin_key, gal_key):
    return ts.hamming(cipher1, cipher2, relin_key, gal_key)'''

