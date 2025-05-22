import tenseal as ts
import os
import stat
import numpy as np
import socket
def read_data(filename):
    with open(filename,'rb') as f:
        filecontent=f.read()
    return filecontent  
def write_data(filename,file_content):
    with open(filename,'wb') as f:
        f.write(file_content)
def cal_dist(enc1,enc2):#计算密文的平方欧氏距离
    euclidean_squared=enc1-enc2
    euclidean_squared=euclidean_squared.dot(euclidean_squared)
    encresu=euclidean_squared.serialize()
    write_data('./recv/result.txt',encresu)
    return euclidean_squared 
def receive_files(host, port):
    save_path='./recv/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 创建套接字
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定地址和端口
        s.bind((host, port))
        # 监听连接
        s.listen()
        print("等待客户端连接...")
        conn, addr = s.accept()
        print(f"连接成功: {addr}")
        while True:
            # 接收文件名
            file_name = conn.recv(1024).decode()
            if file_name=='END' or file_name==0:
                break
            # 接收文件大小
            file_size_bytes = conn.recv(8)
            file_size = int.from_bytes(file_size_bytes, byteorder='big')
            print(file_name,file_size)
            # 接收文件内容
            received_size = 0
            file_data = b''
            while received_size < file_size:
                chunk = conn.recv(2048)
                if not chunk:
                    break
                file_data += chunk
                received_size += len(chunk)
            # 保存接收到的文件
            file_path = save_path+'1.txt'
            write_data(file_path,file_data)
            print(f"接收到文件: {file_name}")
        print("所有文件接收完毕")
        
# 创建 socket 对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定主机和端口
server_address = ('localhost', 8888)
server_socket.bind(server_address)

# 监听连接
server_socket.listen(1)
print("等待客户端连接...")
conn, client_address = server_socket.accept()
print("连接来自:", client_address)
def receive_file2(fsize,n):
    file_data=b''
    while len(file_data)<fsize:
        remain=fsize-len(file_data)
        chunk= conn.recv(min(remain,1024))
        if not chunk:
            break
        file_data+=chunk
    write_data('./recv/%s.txt'%n,file_data)
    if n==1:
        print(f"已接收context并保存")
    elif n==2:
        print(f"已接收密文特征c1并保存")
    else:
        print(f"已接收密文特征c2并保存")
    # 发送确认信息给客户端
    confirmation_message = f"文件已成功接收"
    conn.sendall(confirmation_message.encode())
n=1
# 接收文件1
file_size = int(conn.recv(1024).decode('utf-8'))
receive_file2(file_size,n)
n+=1
# 接收文件2
file_size = int(conn.recv(1024).decode('utf-8'))
receive_file2(file_size,n)
n+=1
# 接收文件3
file_size = int(conn.recv(1024).decode('utf-8'))
receive_file2(file_size,n)
#计算
print('开始计算')
context=ts.context_from(read_data('./recv/1.txt'))
c1=ts.lazy_ckks_vector_from(read_data('./recv/2.txt'))
c1.link_context(context)
c2=ts.lazy_ckks_vector_from(read_data('./recv/3.txt'))
c2.link_context(context)
dist=cal_dist(c1,c2)
fsize=os.path.getsize('./recv/result.txt')
conn.send(str(fsize).encode('utf-8'))
resu=read_data('./recv/result.txt')
conn.sendall(resu)
print('结果发送成功')
conn.close()
server_socket.close()

'''if __name__=='__main__':
    # 创建套接字对象
    #server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定IP地址和端口
    #server.bind(('localhost', 8888))
    # 监听连接
    #server.listen(1)
    #print("等待客户端连接...")
    # 接受客户端连接
    #client_socket, client_address = server.accept()
    #print("客户端已连接：", client_address)
    # 接收数据
    received_data = b""
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        
    receive_files('localhost', 8888)

    #print('已接收到密文和context\n')
    context=ts.context_from(read_data('./recv/202404171525532nv7mASziF84D5i_public.txt'))
    
    context=ts.context_from(ctx)
    c1=ts.lazy_ckks_vector_from(enc1)
    c1.link_context(context)
    c2=ts.lazy_ckks_vector_from(enc2)
    c2.link_context(context)
    dist=cal_dist(c1,c2)
    distance=dist.serialize()
    client_socket.send(distance)

    # 关闭套接字连接
    client_socket.close()
    server.close()'''
