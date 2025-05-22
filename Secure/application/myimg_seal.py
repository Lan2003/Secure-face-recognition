import os
import numpy as np

from myload_encf_todb import get_tFeature
import tenseal as ts


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

#计算两个二维数组之间的余弦相似度
def simil2(f1, f2):#cosine_similarity
    # compute cosine_similarity for 2-D array
    #f1 = f1.numpy()
    #f2 = f2.numpy()

    A = np.sum(f1*f2, axis=0)
    B = np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0) + 1e-5

    return A / B    
def OU(enc1,enc2,context):
    #enc1=ts.ckks_vector_from(context,enc1)
    #enc2=ts.ckks_vector_from(context,enc2)
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    return euclidean_squared

def write_data(filename,file_content):
    with open(filename,'wb') as f:
        f.write(file_content)
        
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent

#客户端，获取特征
#get_Feature1(user_id,'old' ,0, mask=None, binary=False)
#get_Feature1(user_id,'new' ,0, mask=None, binary=False)
'''img1_embedding=read_data('./application/data/originUserFaceData/'+user_id + '.old' + '.txt')
img2_embedding=read_data('./application/data/originUserFaceData/'+user_id + '.new' + '.txt')
img1_embedding_array = np.loadtxt(img1_embedding.decode('utf-8')).tolist()
img2_embedding_array = np.loadtxt(img2_embedding.decode('utf-8')).tolist()'''
#img1_embedding_array= np.loadtxt('./application/data/originUserFaceData/'+user_id + '.old' + '.txt', delimiter=",").tolist()
#img2_embedding_array= np.loadtxt('./application/data/originUserFaceData/'+user_id + '.new' + '.txt', delimiter=",").tolist()
#print('img1_embedding_array',img1_embedding_array)
'''print(img1_embedding_array.shape)#输出向量维度，形状

print(img1_embedding_array.ndim)'''
#初始 客户端
'''context = ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])   
context.global_scale = pow(2, 40)
context.generate_galois_keys()
secret_context=context.serialize(save_secret_key=True)
write_data('D:/Face/Secure/application/seal/secret.txt',file_content=secret_context)
context.make_context_public()
public_context=context.serialize()
write_data(filename='D:/Face/Secure/application/seal/public.txt',file_content=public_context)
'''
'''user_id='20240421224158a6eBCdWkvLm9Jbv'
f1=get_tFeature('202404171525532nv7mASziF84D5i.old.png',0, mask=None, binary=False)
f2=get_tFeature('202404171525532nv7mASziF84D5i.new.png',0, mask=None, binary=False)
#加密
context=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))#需要序列化
enc1=ts.ckks_vector(context,f1)
enc2=ts.ckks_vector(context,f2)
print('enc1',enc1)
enc1_proto=enc1.serialize() #序列化
enc2_proto=enc2.serialize()#序列化
write_data('D:/Face/Secure/application/data/encryptUserFaceData/202404171525532nv7mASziF84D5i.old.txt',enc1_proto)
write_data('D:/Face/Secure/application/data/encryptUserFaceData/202404171525532nv7mASziF84D5i.new.txt',enc2_proto)'''
#计算 服务器端
user_id='20240421224158a6eBCdWkvLm9Jbv'
ctx=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))#
enc1_proto=read_data('./faceset/facedata/20240421224158a6eBCdWkvLm9Jbv.txt')
feat=get_tFeature(user_id+'.old.png',0, mask=None, binary=False)
enc2=ts.ckks_vector(ctx,feat)
#enc2_proto=read_data('./application/data/encryptUserFaceData/20240421224158a6eBCdWkvLm9Jbv.old.txt')

enc1=ts.lazy_ckks_vector_from(enc1_proto)
enc1.link_context(ctx)

'''enc2=ts.lazy_ckks_vector_from(enc2_proto)
enc2.link_context(ctx)'''

#计算欧几里得距离的平方
euclidean_squared=enc1-enc2
euclidean_squared=euclidean_squared.dot(euclidean_squared)


'''
write_data('D:/Face/Secure/application/seal/euclidean_squared.txt',euclidean_squared.serialize())
try:#这里检验服务器计算端不能使用私钥，所以会返回解密错误
    euclidean_squared.decrypt()
except Exception as err:
    print("Exception:",str(err))
print('euclidean_squared:',euclidean_squared)
'''

#解密 客户端？
'''context=ts.context_from(read_data('D:/Face/Secure/application/seal/secret.txt'))
euclidean_squared_proto=read_data('D:/Face/Secure/application/seal/euclidean_squared.txt')
euclidean_squared=ts.lazy_ckks_vector_from(euclidean_squared_proto)
euclidean_squared.link_context(context)'''
euclidean_squared_plain=euclidean_squared.decrypt()[0]
if euclidean_squared_plain<100:
    print('same person')
else:
    print('different person')
simi1=simil(enc1,enc2,ctx)
#simi2=simil2(f1,f2)
ham2=enc1-enc2
print('simi1密文相似度:',simi1)
ham2.link_context(ctx)
ham2=ham2.decrypt()[0]
print('ham2密文相减:',ham2)
#print('simi2明文特征余弦相似度：',simi2)
#验证
#distance=dst.findEuclideanDistance(img1_embedding,img2_embedding)
#print('euclidean_squared-traditional:',distance*distance)
print('euclidean_squared-homomorphic:',euclidean_squared_plain)
#print(abs(distance*distance-euclidean_squared_plain))#小于0.00001?

'''
#以下是一个正常的遍历计算距离
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent
def cal_dist(enc1,enc2):#计算密文的平方欧氏距离
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    return euclidean_squared

folder_path = './faceset/facedata'
min_distance=float('inf')
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是以 .txt 结尾的文本文件
    if filename.endswith('.txt'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        enc3_proto=read_data(file_path)
        enc3=ts.lazy_ckks_vector_from(enc3_proto)
        enc3.link_context(context)
            # 计算欧式距离
        distance = cal_dist(enc1, enc3)
        distance=distance.decrypt()[0]
        print('distance',distance)
            # 更新最小距离和对应的名称
        if distance <= min_distance:
            min_distance = distance
            matching_name = os.path.splitext(filename)[0]
        
    min_dist=abs(min_distance) 
    if min_dist<=0.35:
            # 输出结果
        print("匹配的名称：", matching_name)
        print('平方欧式距离：',min_dist)'''