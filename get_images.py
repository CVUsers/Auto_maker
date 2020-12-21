"""
create by cv调包侠
支持实时标注和视频标注和图片的文件夹标注
github:https://github.com/CVUsers/Auto_maker 欢迎star~
公众号在readme~欢迎加入社区.
"""
import cv2
from os import getcwd
import os
from os import getcwd
from xml.etree import ElementTree as ET
import numpy as np
import argparse
import random


# 定义一个创建一级分支object的函数
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    _object = ET.SubElement(root, 'object')  # 创建一级分支object
    name = ET.SubElement(_object, 'name')  # 创建二级分支
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(_object, 'bndbox')  # 创建bndbox
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 创建xml文件的函数
def create_tree(image_name, imgdir, h, w):
    global annotation
    annotation = ET.Element('annotation')  # 创建树根annotation
    folder = ET.SubElement(annotation, 'folder')  # 创建一级分支folder
    folder.text = (imgdir)  # 添加folder标签内容
    filename = ET.SubElement(annotation, 'filename')  # 创建一级分支filename
    filename.text = image_name
    path = ET.SubElement(annotation, 'path')  # 创建一级分支path
    path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录
    source = ET.SubElement(annotation, 'source')  # 创建一级分支source
    database = ET.SubElement(source, 'database')  # 创建source下的二级分支database
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')  # 创建一级分支size
    width = ET.SubElement(size, 'width')  # 创建size下的二级分支图像的宽、高及depth
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')  # 创建一级分支segmented
    segmented.text = '0'


def find_max_name(classes, mix=False):  # 自动寻找下一张保存图片的名称
    max = 0
    for i in os.listdir('images/'):
        if '_' not in i:
            continue
        elif 'mix' in i and i.split('_')[1] == classes:
            if int(i.split('_')[2].split('.')[0]) > max:
                max = int(i.split('_')[2].split('.')[0])

        elif i.split('_')[0] == classes:
            if int(i.split('_')[1].split('.')[0]) > max:
                max = int(i.split('_')[1].split('.')[0])
    return max


def mix_roi_img(mix, img, x, y, w, h):  # 使用mix_up贴图
    global counter
    if os.path.isdir(mix):
        i = random.choice(os.listdir(mix))
        img_back = cv2.imread(os.path.join(mix, i))
        try:
            img_back = cv2.resize(img_back, (640, 480))
        except:
            print(f'{os.path.join(mix, i)} connot open it!')
        rows, cols, channels = img.shape  # rows，cols最后一定要是前景图片的，后面遍历图片需要用到
        center = [x, y]  # 在新背景图片中的位置
        for i in range(cols):
            for j in range(rows):
                # if dilate[i, j] == 0:
                if center[0] + i < 640 and center[1] + j < 480:
                    img_back[center[1] + j, center[0] + i] = img[j, i]  # 此处替换颜色，为BGR通道
        cv2.imshow(f'mix_{i}', img_back)
        cv2.waitKey(30)
        counter += 1
        if counter % 20 == 0:
            cv2.destroyAllWindows()
        return img_back


def saveROIImg(frame, img, xmin, ymin, xmax, ymax, obj_name, flag=False, mix=False):  # 保存图片和xml
    global counter, saveimg
    name = find_max_name(obj_name, mix)
    H, W = frame.shape[0], frame.shape[-2]
    name += 1
    if flag:
        print("Saving image:", name, xmin, ymin, xmax, ymax)
        cv2.imwrite(path + f'mix_{obj_name}_' + str(name) + ".jpg", mix)
        cv2.rectangle(mix, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(test_path + f'mix_{obj_name}_' + str(name) + ".jpg", mix)
        create_tree(f'mix_{obj_name}_' + str(name) + '.jpg ', 'images', H, W)
        create_object(annotation, xmin, ymin, xmax, ymax, obj_name)
        cv2.waitKey(180)
        tree = ET.ElementTree(annotation)
        tree.write('.\Annotations\{}.xml'.format(f'mix_{obj_name}_' + str(name)))
        return
    print("Saving image:", name, xmin, ymin, xmax, ymax)
    cv2.imwrite(path + f'{obj_name}_' + str(name) + ".jpg", img)
    cv2.imwrite(test_path + f'{obj_name}_' + str(name) + ".jpg", frame)
    cv2.imshow('images', img)
    create_tree(f'{obj_name}_' + str(name) + '.jpg ', 'images', H, W)
    create_object(annotation, xmin, ymin, xmax, ymax, obj_name)
    cv2.waitKey(50)
    tree = ET.ElementTree(annotation)
    tree.write('.\Annotations\{}.xml'.format(f'{obj_name}_' + str(name)))


def run_on_video(source, mix=False):  # 视频与实时标注入口
    saveimg = False
    mix_img = False
    wd = getcwd()
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()  # 定义追踪器
    intBB = None
    vs = cv2.VideoCapture(source)
    while True:
        frame = vs.read()
        frame = frame[1]
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 3)
        frame1 = frame.copy()
        (H, W) = frame.shape[:2]
        if frame is None:
            break
        if intBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                if mix:
                    Scaling_probability = random.randint(args.Scaling_probability[0] * 10,
                                                         args.Scaling_probability[1] * 10) / 10
                    try:
                        mix_frame = cv2.resize(frame1[y:y + h, x:x + w],
                                               (int(w * Scaling_probability), int(h * Scaling_probability)))
                        w_, h_ = int(w * Scaling_probability), int(h * Scaling_probability)
                        mix_img = mix_roi_img(mix, mix_frame, x, y, w_, h_)
                        if saveimg:
                            saveROIImg(frame, frame1, x, y, x + w_, y + h_, obj_name, flag=True, mix=mix_img)
                    except:
                        pass
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if saveimg:
            saveROIImg(frame, frame1, x, y, x + w, y + h, obj_name)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('s'):
            print('class is:', obj_name)
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
            tracker1 = OPENCV_OBJECT_TRACKERS[args.tracker]()
            intBB = None
            intBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, intBB)
        elif key == ord('b'):
            saveimg = True
        elif key == 27:
            cv2.destroyAllWindows()
            vs.release()
            break


def run_on_images(source, mix=False):
    saveimg = False
    wd = getcwd()
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
    intBB = None
    for i in os.listdir(source):
        frame = cv2.imread(os.path.join(source, i))
        print(os.path.join(source, i))
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 3)
        frame1 = frame.copy()
        (H, W) = frame.shape[:2]
        if frame is None:
            break
        if intBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                if mix:
                    Scaling_probability = random.randint(args.Scaling_probability[0] * 10,
                                                         args.Scaling_probability[1] * 10) / 10
                    try:
                        mix_frame = cv2.resize(frame1[y:y + h, x:x + w],
                                               (int(w * Scaling_probability), int(h * Scaling_probability)))
                        w_, h_ = int(w * Scaling_probability), int(h * Scaling_probability)
                        mix_img = mix_roi_img(mix, mix_frame, x, y, w_, h_)
                        if saveimg:
                            saveROIImg(frame, frame1, x, y, x + w_, y + h_, obj_name, flag=True, mix=mix_img)
                    except:
                        pass
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if saveimg:
            saveROIImg(frame, frame1, x, y, x + w, y + h, obj_name)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('s'):
            print('class is:', obj_name)
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
            tracker1 = OPENCV_OBJECT_TRACKERS[args.tracker]()
            intBB = None
            intBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, intBB)
        elif key == ord('b'):
            saveimg = True
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    """
    cv调包侠
    注意：为了更好的效果，您在开始使用后，先按下s键开始标注，按下b键开始保存，
    期间如果有不准确的地方，您有几个机会去调整boxes：1 再按下s键，重新描框即可；2您可以通过test_img文件夹重新筛选删除图片。
    """
    parser = argparse.ArgumentParser('Auto_maker')
    parser.add_argument('-t', "--tracker", type=str, default='csrt', help='choose opencv tracker methods')
    parser.add_argument('-i', '--source_path', type=str, default='0',
                        help='0 or 1 is your capture, or use video.mp4 or use path_dir like: ./images/')
    parser.add_argument('--show', default=True, action='store_true', help='mix_up picture show')
    parser.add_argument('--mix', default='./mix_img/', action='store_true',
                        help='default:False is do not use mix_up method, and use ./mix_up to mix_up')
    parser.add_argument('--Scaling_probability', default=[0.6, 1.4], action='store_true',
                        help='The probability of scaling your boxes')
    classes_list = ['UsingPhone', 'LikePhone']  # 类别名称 不建议使用"_"命名
    obj_name = classes_list[0]  # 此次标注的类别名称（注意修改此处）
    args = parser.parse_args()
    counter, flag = 0, 0
    path = "images/"
    test_path = 'test_img/'
    OPENCV_OBJECT_TRACKERS = {  # OPENCV_OBJECT_TRACKERS 默认使用csrt
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerKCF_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    if os.path.isdir(args.source_path):  # 图片文件夹自动标注
        run_on_images(args.source_path, mix=args.mix)
    elif os.path.isfile(args.source_path):  # 标注一个视频文件
        run_on_video(args.source_path, mix=args.mix)
    elif '0' in args.source_path or '1' in args.source_path:  # 实时标注 （建议使用实际使用时的相机录制~）
        run_on_video(int(args.source_path), mix=args.mix)
