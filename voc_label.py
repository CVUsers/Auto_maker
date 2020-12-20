import os
import xml.etree.ElementTree as ET
import os
import cv2
import random
classes = ['UsingPhone', 'LikePhone']
def convert(size, box):
    print(size, box)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    print(image_id)
    in_file = open(r'./Annotations/%s' % (image_id), 'rb')  #  读取xml文件路径
    out_file = open('./labels/%s.txt' % (image_id.split('.')[0]), 'w')  #  需要保存的txt格式文件路径
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if w == 0 and h == 0:
        img = cv2.imread('./images/' +image_id.replace('xml', 'jpg'))
        w, h = img.shape[1], img.shape[0]
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print('*******************************'*2, cls)
            break
        cls_id = classes.index(cls)
        if cls == 'A' or cls == 'E' or cls == 'M' or cls == 'N' or cls == 'S' or cls == 'T' or cls =='F':
            cls_id = 0
        if  cls == 'L' or cls =='U'  or cls == 'H':
            cls_id = 1
        if cls == 'Y':
            cls_id = 2
        if cls == 'I' or cls == 'J':
            cls_id = 3
        if cls == 'G' or cls == 'Q' or cls =='P':
            cls_id = 4
        if cls == 'O' or cls == 'C':
            cls_id = 5
        if cls == 'K' or cls == 'V' or cls == 'W':
            cls_id = 6
        if cls == 'X' or cls == 'Z' or cls == 'R' or cls == 'D':
            cls_id = 7
        if cls_id == 26 or cls == 'B':
            cls_id =8
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


image_ids_train = os.listdir('./Annotations')  # 读取xml文件名索引

for image_id in image_ids_train:
    print(image_id)
    convert_annotation(image_id)


trainval_percent = 0.1  # 可自行进行调节
train_percent = 1
xmlfilepath = './labels'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftest = open('./test.txt', 'w')
ftrain = open('./train.txt', 'w')

for i in list:
    name = total_xml[i] + '\n'
    if i in trainval:
        if i in train:
            ftest.write('../images/' + name.replace('txt', 'jpg'))
    else:
        ftrain.write('../images/' + name.replace('txt', 'jpg'))
ftrain.close()
ftest.close()
