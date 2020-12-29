"""
create by cv调包侠
支持实时标注和视频标注和图片的文件夹标注
github:https://github.com/CVUsers/Auto_maker 欢迎star~
公众号在readme~欢迎加入社区.
"""
import sys

import cv2
from os import getcwd
import os
from os import getcwd
from xml.etree import ElementTree as ET
import numpy as np
import argparse
import random


# pip install opencv-contrib-python

# 定义一个创建一级分支object的函数
def create_object(root, xi=None, yi=None, xa=None, ya=None, obj_name=None, mutils=None):  # 参数依次，树根，xmin，ymin，xmax，ymax
    if mutils:
        for index, xyxy in enumerate(mutils):
            _object = ET.SubElement(root, 'object')  # 创建一级分支object
            name = ET.SubElement(_object, 'name')  # 创建二级分支
            name.text = str(obj_name[index])
            pose = ET.SubElement(_object, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(_object, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(_object, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(_object, 'bndbox')  # 创建bndbox
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = '%s' % xyxy[0]
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = '%s' % xyxy[1]
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = '%s' % xyxy[2]
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = '%s' % xyxy[3]
    else:
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


# 自动寻找下一张保存图片的名称
def find_max_name(classes, mix=False):
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


# 使用mix_up贴图
def mix_roi_img(mix, img, x=None, y=None, w=None, h=None, mutil_mix=False, mix_xyxy=None):
    global counter
    if os.path.isdir(mix) and not mutil_mix:
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
                if center[0] + i < 640 and center[1] + j < 480:
                    img_back[center[1] + j, center[0] + i] = img[j, i]  # 此处替换颜色，为BGR通道
        cv2.imshow(f'mix_{i}', img_back)
        cv2.waitKey(20)
        counter += 1
        if counter % 20 == 0:
            cv2.destroyAllWindows()
        return img_back
    elif os.path.isdir(mix) and mutil_mix:
        i = random.choice(os.listdir(mix))
        img_back = cv2.imread(os.path.join(mix, i))
        try:
            img_back = cv2.resize(img_back, (640, 480))
        except:
            print(f'{os.path.join(mix, i)} connot open it!')
        for index, img_single in enumerate(img):
            rows, cols, channels = img_single.shape  # rows，cols最后一定要是前景图片的，后面遍历图片需要用到
            center = [mix_xyxy[index][0], mix_xyxy[index][1]]  # 在新背景图片中的位置
            for i in range(cols):
                for j in range(rows):
                    if center[0] + i < 640 and center[1] + j < 480:
                        img_back[center[1] + j, center[0] + i] = img_single[j, i]  # 此处替换颜色，为BGR通道
        cv2.imshow(f'mix_{i}', img_back)
        cv2.waitKey(20)
        counter += 1
        if counter % 20 == 0:
            cv2.destroyAllWindows()
        return img_back


# 保存图片和xml
def saveROIImg(frame, img, xmin, ymin, xmax, ymax, obj_name, flag=False, mix=False):
    global counter, saveimg
    name = find_max_name(obj_name, mix)
    H, W = frame.shape[0], frame.shape[-2]
    if flag:
        name += 1
        print("Saving image:", f'mix_{obj_name}_' + str(name) + ".jpg", xmin, ymin, xmax, ymax)
        cv2.imwrite(path + f'mix_{obj_name}_' + str(name) + ".jpg", mix)
        cv2.rectangle(mix, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(test_path + f'mix_{obj_name}_' + str(name) + ".jpg", mix)
        create_tree(f'mix_{obj_name}_' + str(name) + '.jpg ', 'images', H, W)
        create_object(annotation, xmin, ymin, xmax, ymax, obj_name)
        cv2.waitKey(50)
        tree = ET.ElementTree(annotation)
        tree.write('.\Annotations\{}.xml'.format(f'mix_{obj_name}_' + str(name)))
        return
    name += 1
    print("Saving image:", f'{obj_name}_' + str(name) + ".jpg", xmin, ymin, xmax, ymax)
    cv2.imwrite(path + f'{obj_name}_' + str(name) + ".jpg", img)
    cv2.imwrite(test_path + f'{obj_name}_' + str(name) + ".jpg", frame)
    cv2.imshow('images', img)
    create_tree(f'{obj_name}_' + str(name) + '.jpg ', 'images', H, W)
    create_object(annotation, xmin, ymin, xmax, ymax, obj_name)
    cv2.waitKey(20)
    tree = ET.ElementTree(annotation)
    tree.write('.\Annotations\{}.xml'.format(f'{obj_name}_' + str(name)))


# 视频与实时标注入口（单类别）
def run_on_video(source, mix=False):
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
                    mix_frame = cv2.resize(frame1[y:y + h, x:x + w],
                                           (int(w * Scaling_probability), int(h * Scaling_probability)))
                    w_, h_ = int(w * Scaling_probability), int(h * Scaling_probability)
                    mix_img = mix_roi_img(mix, mix_frame, x, y, w_, h_)
                    if saveimg:
                        saveROIImg(frame, frame1, x, y, x + w_, y + h_, obj_name, flag=True, mix=mix_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if saveimg:
            saveROIImg(frame, frame1, x, y, x + w, y + h, obj_name)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
            intBB = None
            intBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            print('now class is:', obj_name)
            tracker.init(frame, intBB)
        elif key == ord('b'):
            saveimg = True
        elif key == 27:
            cv2.destroyAllWindows()
            vs.release()
            break


# 图片单类别标注
def run_on_images(source, mix=False):
    saveimg = False
    wd = getcwd()
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
    intBB = None
    intBB1 = None
    for i in os.listdir(source):
        frame = cv2.imread(os.path.join(source, i))
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
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
            tracker1 = OPENCV_OBJECT_TRACKERS[args.tracker]()
            intBB = None
            intBB1 = None
            intBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, intBB)
            intBB1 = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            tracker1.init(frame, intBB1)
        elif key == ord('b'):
            saveimg = True
        elif key == 27:
            cv2.destroyAllWindows()
            break


# 保存多类别图片和xml
def saveMutilImg(frame, img, xyxy, obj_name, flag=False, mix=False):
    global counter, saveimg
    name = find_max_name(''.join(obj_name), mix)
    H, W = frame.shape[0], frame.shape[-2]
    if flag:
        name += 1
        print("Saving mixed_image:", ''.join(obj_name) + '_' + str(name), xyxy)
        cv2.imwrite(path + 'mix_' + ''.join(obj_name) + '_' + str(name) + ".jpg", mix)
        for rectangle in xyxy:
            cv2.rectangle(mix, (rectangle[0], rectangle[1]), (rectangle[0], rectangle[3]), (0, 255, 0), 2)
        cv2.imwrite(test_path + 'mix_' + ''.join(obj_name) + '_' + str(name) + ".jpg", mix)
        create_tree('mix_' + ''.join(obj_name) + '_' + str(name) + '.jpg ', 'images', H, W)
        create_object(annotation, mutils=xyxy, obj_name=obj_name)
        cv2.waitKey(20)
        tree = ET.ElementTree(annotation)
        tree.write('.\Annotations\{}.xml'.format('mix_' + ''.join(obj_name) + '_' + str(name)))
        return
    name += 1
    print("Saving mutil_image:", ''.join(obj_name) + '_' + str(name), xyxy)
    cv2.imwrite(path + ''.join(obj_name) + '_' + str(name) + ".jpg", img)
    cv2.imwrite(test_path + ''.join(obj_name) + '_' + str(name) + ".jpg", frame)
    cv2.imshow('images', img)
    create_tree(''.join(obj_name) + '_' + str(name) + '.jpg ', 'images', H, W)
    create_object(annotation, mutils=xyxy, obj_name=obj_name)
    cv2.waitKey(20)
    tree = ET.ElementTree(annotation)
    tree.write('.\Annotations\{}.xml'.format(''.join(obj_name) + '_' + str(name)))


# 多类别标注入口
def mutil_labels_video(source, mix=False, max_number=10):
    saveimg = False
    mix_img = False
    track_flag_list = [None] * max_number
    tracker_list = [None] * max_number
    label_list = [''] * max_number
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
        xyxy = []
        mix_xyxy = []
        mix_frames = []
        for index, track in enumerate(track_flag_list):
            if track is not None:
                (success, box) = tracker_list[index].update(frame)
                # if success:
                (x, y, w, h) = [int(v) for v in box]
                xyxy.append((x, y, x + w, y + h))
                if mix:
                    Scaling_probability = random.randint(args.Scaling_probability[0] * 10,
                                                         args.Scaling_probability[1] * 10) / 10
                    mix_frame = cv2.resize(frame1[y:y + h, x:x + w],
                                           (int(w * Scaling_probability), int(h * Scaling_probability)))
                    mix_frames.append(mix_frame)
                    w_, h_ = int(w * Scaling_probability), int(h * Scaling_probability)
                    mix_xyxy.append((x, y, w_, h_))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if mix_frames:
            mix_imgs = mix_roi_img(mix, img=mix_frames, mix_xyxy=mix_xyxy, mutil_mix=True)
        if saveimg and mix:
            saveMutilImg(frame, frame1, xyxy=xyxy, obj_name=label_list, flag=True, mix=mix_imgs)
        if saveimg:
            saveMutilImg(frame, frame1, xyxy=xyxy, obj_name=label_list)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            track_flag_list = [None] * max_number
            tracker_list = [None] * max_number
            label_list = [''] * max_number
            try:
                now_labels = int(input('请输入现在图片中要标注的类别数目(回车结束)：'))
            except:
                print('请输入数字，如：2')
                sys.exit()
            for label in range(now_labels):
                tracker_list[label] = OPENCV_OBJECT_TRACKERS[args.tracker]()
                track_flag_list[label] = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
                label_list[label] = str(input('请输入目前这个框的物体类别(回车结束):'))
                print('now class is:', label_list[label])
                tracker_list[label].init(frame, track_flag_list[label])
        elif key == ord('b'):
            saveimg = True
        elif key == 27:
            cv2.destroyAllWindows()
            vs.release()
            break


if __name__ == '__main__':
    """
    cv调包侠 Auto_maker使用教程
    注意：为了更好的效果，您在开始使用后，先按下s键开始标注，按下b键开始保存，期间如果有不准确的地方，您有几个机会去调整boxes：
    1 再按下s键，重新描框即可；
    2 您可以通过test_img文件夹重新筛选删除图片。
    """
    parser = argparse.ArgumentParser('Auto_maker')
    parser.add_argument('-t', "--tracker", type=str, default='csrt', help='choose opencv tracker methods， 选择追踪方法')
    parser.add_argument('-i', '--source_path', type=str, default='0',
                        help='0 or 1 is your capture, obbr use video.mp4 or use path_dir like: ./images/'
                             '0/1使用相机，图像文件夹路径和视频路径都可以')
    parser.add_argument('--show', default=True, help='mix_up picture show, 展示')
    parser.add_argument('--mix', type=str, default=False,
                        help='default:False is do not use mix_up method, and use ./mix_img to mix_up, '
                             '默认为False 则会不使用mix_up数据增强策略~使用：./mix_img或其他图像路径去mixup')
    parser.add_argument('--Scaling_probability', default=[0.6, 1.4], action='store_true',
                        help='The probability of scaling your boxes,设置mix——up时的图像随机等比例缩放范围')
    parser.add_argument('--multi_cls', default=True,
                        help='You can define how many trackers to start,设置为True则能够实时标注多目标')
    """
    cv调包侠 更行 2020.12.27 超参数使用说明（必看）
    如果您是使用单类别标注：请设置--multi_cls 的default 为False
    开启multi_cls也可以单类别标注
    """
    classes_list = ['UsingPhone', 'LikePhone']  # 类别名称 不建议使用"_"命名（如果选择多类别标注，可以不用写）
    obj_name = classes_list[0]  # 此次标注的类别名称（注意修改此处）
    args = parser.parse_args()
    counter, flag = 0, 0
    path = "images/"
    test_path = 'test_img/'
    OPENCV_OBJECT_TRACKERS = {  # 追踪方法 默认使用csrt
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    if os.path.isdir(args.source_path):  # 图片文件夹自动标注
        run_on_images(args.source_path, mix=args.mix)
    elif os.path.isfile(args.source_path) and not args.multi_cls:  # 标注一个视频文件
        run_on_video(args.source_path, mix=args.mix)
    elif ('0' in args.source_path or '1' in args.source_path) and not args.multi_cls:  # 实时标注(一个类别) （建议使用实际使用时的相机录制~）
        run_on_video(int(args.source_path), mix=args.mix)
    elif ('0' in args.source_path or '1' in args.source_path or os.path.isfile(
            args.source_path)) and args.multi_cls:  # 实时标注(多个类别)
        if '0' in args.source_path or '1' in args.source_path:
            mutil_labels_video(int(args.source_path), mix=args.mix)
        elif os.path.isfile(args.source_path):
            mutil_labels_video(args.source_path, mix=args.mix)
