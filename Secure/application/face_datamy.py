# -*- coding: utf-8 -*-
import torch
import os
import copy
from PIL import Image
import time
import argparse
import numpy as np
from scipy import misc #删了，改成下一行
import imageio
from application.fpn_model import LResNet50E_IR_Occ as LResNet50E_IR_FPN
import tensorflow as tf
#from application.util import data_encrypt
from application.face.src.align import detect_face#改 删application,单个运行，总体要加回来
from application.face.src.facenet import get_model_filenames, prewhiten#改 删application,单个运行，总体要加回来
from torchvision import transforms
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#model_path = './application/face/src/20180408-102900'
model_path = './application/face/src/modelp5/model_p5.pth.tar'
session = tf.Session()#创建会话
#加载模型
#model=LResNet50E_IR_FPN()
'''
#获取网格数
def get_grids(H, W, N):
    grid_ori = np.zeros((H, W))

    x_axis = np.linspace(0, W, N+1, True, dtype=int)
    y_axis = np.linspace(0, H, N+1, True, dtype=int)

    vertex_set = []
    for y in y_axis:
        for x in x_axis:
            vertex_set.append((y, x))

    grids = [grid_ori]
    for start in vertex_set:
        for end in vertex_set:
            if end[0] > start[0] and end[1] > start[1]:
                grid = grid_ori.copy()
                grid[start[0]:end[0], start[1]:end[1]] = 1.0
                grids.append(grid)
    return grids'''
#image_size=(112,96)
#pattern = int(model_path[model_path.find('p')+1])
#num_mask = len(get_grids(112,96,5))
#print(num_mask)
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

def parse_arguments(argv):#解析命令行参数，返回参数对象
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=[112, 96])
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
def load_and_align_data(image_path, image_sizeh,image_sizew ,margin, gpu_memory_fraction):
    minsize = 20  # 人脸最小尺寸
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor缩放因子
    #MTCNN模型初始化
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)#加载模型，人脸检测+对齐
    
    img = imageio.imread(os.path.expanduser(image_path), pilmode='RGB')#misc.imread改成imageio.加载图像
    img_size = np.asarray(img.shape)[0:2]#获取图像的高度和宽度保存在numpy数组中
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)#进行人脸检测，获取人脸边界框

    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]#裁剪
    aligned = misc.imresize(cropped, (image_sizeh, image_sizew), interp='bilinear')#将裁剪后的图像调整为指定大小
    #print("图像形状:", aligned.shape)
    pre_pil=Image.fromarray(aligned)
    #pre_pil.show()
    return pre_pil
def get_Feature1(img_filename,type ,is_gray, mask=None, binary=False):
    origin_data_path = './application/data/originUserFaceData/'
    img_path = './upload/' + img_filename + '.' + type + '.png'

    time_start = time.time()
    print('开始处理:' + img_filename + '.' + type)
    img=load_and_align_data(img_path, 112,96, 44, 1.0)
    #with open(img_path, 'rb') as f:
        #img =  Image.open(f).convert('RGB')
    img=preprocess(img).unsqueeze(0)
    fc_mask, mask, vec, fc = model(img)

    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    #np.set_printoptions(precision=8, suppress=True)  # 设置小数位数为4位，禁用科学计数法
    fc_mask_array = fc_mask.detach().numpy()
    fc_mask_array = (2 * (fc_mask_array - fc_mask_array.min()) / (fc_mask_array.max() - fc_mask_array.min()) - 1)/10

    #print('fc_mask_array:',fc_mask_array)
    np.savetxt(origin_data_path + img_filename + '.' + type + '.txt', fc_mask_array)#保存
    time_end = time.time()
    print('处理结束:' + img_filename + '.' + type + ',耗时:', end='')
    print('%.2f' % (time_end - time_start))
    if os.access(origin_data_path + img_filename + '.' + type + '.txt', os.F_OK):
        return True
    return False


if __name__ == '__main__':
    get_Feature1('3', 'old', 0, mask=None, binary=False)
    #data_encrypt('20240315192430bGr99xHhqTw8ky2', 'old')
'''
    
#加载和对齐人脸图像数据
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = imageio.imread(os.path.expanduser(image), pilmode='RGB')#misc.imread改成imageio.
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def get_user_data(filename, type):
    origin_data_path = './application/data/originUserFaceData/'
    img_path = './upload/' + filename + '.' + type + '.png'
    args = parse_arguments([img_path])
    time_start = time.time()
    print('开始处理:' + filename + '.' + type)
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with graph.as_default():
        with session.as_default() as sess:
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            np.savetxt(origin_data_path + filename + '.' + type + '.txt', emb[0])
    time_end = time.time()
    print('处理结束:' + filename + '.' + type + ',耗时:', end='')
    print('%.2f' % (time_end - time_start))
    if os.access(origin_data_path + filename + '.' + type + '.txt', os.F_OK):
        return True
    return False
'''
