## 【效率提高10倍项目原创发布！】深度学习数据自动标注器开源 目标检测和图像分类（高精度高效率）


数据标注费时费力，又费钱！深谙其苦的我开发了这个项目。（如果想快速使用请看前半部分）
大家好，我是大家的好朋友~ cv调包侠，深度学习算法攻城狮（实习僧）一枚， 下面我将诚心地发布一个自己的原创：Auto_maker! 

他能干什么？

大家可以特别方便地通过我的Auto_maker 实现目标检测数据集的实时制作，包括：10分钟完成 真实数据采集，自动标注，转换，增强，并且可以直接进行yolov3， yolov4 ，yolov5，efficientdet等，并且可以直接导出成onnx，并使用openvino和tensorRT加速；除了检测以外，还支持分类算法，可以一分钟完成图片智能分类欢迎star~

同时他具有高精度，高实时性，高效率，他是人工标注的10倍以上效率，并且精度可控~

cv调包侠录制了一个视频讲解~大家也可以通过这篇文章得到更多的了解！

注意：为了更好的效果，您在开始使用后，先按下s键开始标注，按下b键开始保存，期间如果有不准确的地方，您有几个机会去调整boxes：1 再按下s键，重新描框即可；2您可以通过test_img 重新筛选删除图片。

github：https://github.com/CVUsers/Auto_maker

项目比较简单，cv调包侠不到半小时就把代码下面开始讲解项目结构和代码。剩下的就是在完善逻辑，优化用户体验了，改了许久，大家可以方便地使用，简单地体验~

## 项目结构与使用教程

### 目标检测模式

![1608460483759](Auto_maker_readme\1.png)

在我们运行Auto_maker 前，需要安装opencv的库：opencv-contrib-python 库

pip install opencv-contrib-python

**然后运行 get_images.py 就能看到实时图像，再按下"s"键就可以用鼠标绘制目标框，绘制完后回车一下~**

**然后按下“b”键就会看到控制台输出开始保存的提示~**

然后我们可以左右上下地平移物体，如果内外前后地移动了物体后，追踪框若是发生了偏移，那么就请再按一下“s”重新标注一下~会继续保存图片到images文件夹中，同时也会生成xml到Annotations文件夹中。

就这样，大家不断地平移，切换场景，并且打开mix_up模式，会获得更多，更丰富的图片，这样对我们训练的模型也会更准确和更泛化。

尤其是目标检测中的多尺度问题，需要我们丰富数据及其标注框在图片中的相对大小来解决，我在mix_up 的同时使用了随机等比例缩放，获得更多样式~。

```python
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
    obj_name = classes_list[0]  # 此次标注的类别名称
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
```

像这样，tracker可以切换成：csrt， kcf，boosting等方式，这是opencv中的追踪算法，csrt是较准的，同时你也可以使用deepsort 进行跟踪，或者使用自己训练好的一个模型，进行其他大量数据的预训练。

--source_path 我们可以切换为0 ：使用内置相机，切换为1：使用外界相机； 切换为图片路径：images/ 下的图片，可以这样标注~，切换为视频路径：demo.mp4 来标注视频帧，注意一个视频帧数很多，大家可以修改程序中的cv2.waitkey()来改善。

--show 就是显示我们的mix_up 的图片~

--mix 是我们使用mix_up 并且使用随机等比例缩放的路径,如果default = False,就是不使用mix_up 做增强，如果使用，就将mix_img 的路径放入：./mix_img/

--Scaling_probability 就是缩放比例的区间。

classes_list ：我们将所有的类别写进来，并

### 影像分类模式



![1608462055629](Auto_maker_readme\2.png)

影像分类中，使用简单的固定ROI方式，在运行maker_classification.py 后我们可以轻松地按下s键保存图片~图片就会根据main中的类别名称保存到data/train/类别名/下面，如果是测试集，就在main中修改为test模式

```python
parser.add_argument('--dtype', type=str, default='pause', help='your label')
parser.add_argument('--train_test', type=str, default='test', help='train/test')
```

--dtype:类别 修改成自己的数据类别，就会在data/train/下面生成这样的文件夹，里面存放这个类别的图片

--train_test  现在收集的数据是训练集还是测试集。切换train或者test会分别保存到train或者test文件夹下。





## 数据采集演示与训练出来的模型演示



github：https://github.com/CVUsers/Auto_maker

![1608466712852](Auto_maker_readme\3.png)

github图片



![1608466791841](Auto_maker_readme\demo1.png)

 																	数据标注部分





![1608467114587](Auto_maker_readme\demo2.png)

​																				自动标注过程



![1608467265739](Auto_maker_readme\4.png)

​																							数据效果图	

![1608467331409](Auto_maker_readme\5.png)

​																				标注文件集

![1608467378947](Auto_maker_readme\6.png)

### 训练出来的目标检测模型演示

模型已放在github上，轻量级模型，这两天会更新更高精度模型~

![1608467474374](D:\CSDN\pic_new\Auto_maker\1608467474374.png)



### 训练出来的分类模型演示【固定框检测模式】

模型已放在github上，轻量级模型，这两天会更新更高精度模型~

![1608467536580](Auto_maker_readme\9.png)

![1608467558431](Auto_maker_readme\8.png)

## 一键训练YOLOv3 YOLOv4 YOLOv5 方法



### 转换数据

我们现在得到了所有的图片/标注文件（同名），那么就可以开始训练了，训练过程很简单，我们只需要转换一下数据：

运行voc_label.py 数据转换成YOLO格式：通过这个脚本，你可以在labels文件夹中生成归一化后的标签，同时生成一份训练集：train.txt 和测试集test.txt

我们只需要修改classes：类别即可。

```python
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

```

### 训练与检测

### 训练

下面YOLOv5 和V3 一样，我们只需要修改yolov5/data/voc.yaml即可（v3 和v4 若是使用darknet也是差不多哦~）：

```python
train: ../train.txt  # 16551 images
val: ../test.txt  # 4952 images

# number of classes
nc: 2

# class names
names: ['UsingPhone', 'LikePhone']
```

在训练此yolov5 前，请安装pytorch1.6 以上，如果你使用的是torch1.5以下，那么请区clone yolov5 的第二个版本以下，或者你使用yolov3~

然后修改train.py 的这个部分

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/voc.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=10, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
```

--weights 预训练模型路径

--cfg 网络结构路径

--data voc.yaml路径

训练效果图可以在runs 下查看result.txt

或者使用tensorboard查看，我们到yolov5或者3 路径下执行tensorboard --logdir=runs



![1608469384253](Auto_maker_readme\10.png)

mAP和precision 和recall 如下，我的模型只训练了70次~ 我接下来会使用4w张图片训练完，来查看准确率，并且实际体验效果，然后我会放在我的github和公众号：70次效果也不错~



这是损失~

![1608469481705](Auto_maker_readme\11.png)

### 检测

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'D:\cvuser\Auto_maker\yolov5\runs\train\exp7\weights\best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', default=True, help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
```

我们把--weights 改成模型的绝对路径即可~

并且打开--augment 为True



## 核心部分介绍

### 目标检测数据标注

在目标检测数据标注代码：get_images.py中：

```python
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
    obj_name = classes_list[0]  # 此次标注的类别名称
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
```

程序入口，判断传入的是文件夹还是视频还是相机路径，做出相应响应。

run_on_video 函数

```python
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
```

先通过tracker = args.tracker ()  定义追踪器，然后显示实时视频，监听鼠标，若为“s” 那么启动追踪器，并获取关键区域roi。同时初始化追踪器。然后获取roi的bounding box位置

```python
 intBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
 tracker.init(frame, intBB)
```

在mix_up 中：

```python
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
```

默认640x480 的图片大小，大家可以修改，然后把图片进行随机贴合。大家可以在这里做更多的贴图算法优化，我这里就是像素点的转换，还有其他方法，比如边缘检测，将需要的部分留下，不需要的部分用原mix_up 的图片替换~

```python

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
```

这边是xml树的构建。

```python
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
```

保存模式，选择是否保存mix_up的图片。

### 分类

maker_classification.py 

```python
import argparse

import win32api
import win32con
import cv2 as cv
import os
import numpy as np
save_path = 'data'

def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1+2:y2, x1+2:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    return dst


def get_data(dtype):
    max = 0
    for i in os.listdir('data/{}/{}'.format(args.train_test, dtype)):
        if int(i.split('_')[2].split('.')[0]) > max:
            max = int(i.split('_')[2].split('.')[0])

    return max + 1

def main():
    if not os.path.isdir('./data/'):
        os.makedirs('./data/')
    if not os.path.isdir('./data/train/'):
        os.makedirs('./data/train/')
    if not os.path.isdir('./data/test/'):
        os.makedirs('./data/test/')
    if not os.path.isdir('./data/train/{}'.format(args.dtype)):
        os.makedirs('./data/train/{}'.format(args.dtype))
    if not os.path.isdir('./data/test/{}'.format(args.dtype)):
        os.makedirs('./data/test/{}'.format(args.dtype))
    m_0 = get_data(args.dtype)
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        roi = get_roi(frame, 100, 350, 100, 350)
        k = cv.waitKey(20)
        if k == 27:  # 按下ESC退出
            break
        elif k == ord('s'):  # 按下'A'会保存当前图片到指定目录下

            cv.imwrite("{}/{}/{}/{}.jpg".format(save_path, args.train_test, args.dtype, m_0), roi)
            m_0 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\0\\%s.jpg" % m_0, flip_image)
            # m_0 += 1
            print('正在保存0-roi图片,本次图片数量:', m_0)
        cv.imshow("roi", roi)
        cv.imshow("frame", frame)
        c = cv.waitKey(20)
        if c == 27:
            break
    cv.waitKey(0)
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='pause', help='your label')
    parser.add_argument('--train_test', type=str, default='test', help='train/test')

    args = parser.parse_args()
    main()

```

maker_by_Guss.py 

```python
import cv2
import imutils
import numpy as np
import argparse
import os

bg = None


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main(dtype):
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 90, 380, 285, 590

    num_frames = 0
    thresholded = None

    count = 0

    while(True):
        (grabbed, frame) = camera.read()
        if grabbed:

            frame = imutils.resize(frame, width=700)

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                hand = segment(gray)

                if hand is not None:
                    (thresholded, segmented) = hand

                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1

            cv2.imshow('Video Feed', clone)
            if not thresholded is None:
                cv2.imshow('Thesholded', thresholded)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord('q'):
                break

            if keypress == ord('s'):
                if not os.path.isdir('./data/'):
                    os.makedirs('./data/')
                if not os.path.isdir('./data/train/'):
                    os.makedirs('./data/train/')
                if not os.path.isdir('./data/test/'):
                    os.makedirs('./data/test/')
                if not os.path.isdir('./data/train/{}'.format(args.dtype)):
                    os.makedirs('./data/train/{}'.format(args.dtype))
                if not os.path.isdir('./data/test/{}'.format(args.dtype)):
                    os.makedirs('./data/test/{}'.format(args.dtype))
                cv2.imwrite('data/{}/saved_v2_{:04}.jpg'.format(dtype, count), thresholded)
                count += 1
                print(count, 'saved.')

        else:
            camera.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='pause', help='your label')
    args = parser.parse_args()
    main(args.dtype)
    cv2.destroyAllWindows()

```

使用高斯边缘消除后保存，适用于特征鲜明物体。



### 分类训练部分

```python
class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = mobilenet_v2(pretrained=True) #     backbone + neck + head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(1280, len(args.classes)) # [bs, 1280] -> [bs, classes]

    def forward(self, x): # [bs,3,224,224]
        x = self.net.features(x) # [bs, 1280, 7, 7]  224//32
        x = self.avg_pool(x) # [bs, 1280, 1, 1]
        x = x.view(x.size(0), -1) # [bs, 1280]
        # x = torch.reshape()
        x = self.logit(x)
        return x
```

定义网络和主干网络

```python
def run(images_list, val_list):
    train_dataset = GestureDataset(images_list)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataset = GestureDataset(val_list)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    model = Net()

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_score = float("inf") # 0XFFFFFFF
    best_acc = 0.
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        train_one(train_dataloader, model, optimizer, loss_fn, None)
        scores = val_one(val_dataloader, model, loss_fn)
        if scores['loss'] <= best_score:
            best_score = scores['loss']
            print('*****best_loss:', best_score, 'acc:', best_acc)
            if scores['accuracy'] >= best_acc:
                best_acc = scores['accuracy']
                print('*******save best*******', epoch)
                torch.save(model.state_dict(), "ckpt/model.pth")
```

训练部分

```python
class GestureDataset(Dataset):
    def __init__(self, images_list, transformers=None):
        self.images_list = images_list # 3000
        self.transformers = transformers
    def __len__(self):
        return len(self.images_list)
    def normalize(self, image):
        image = np.transpose(image, (2, 0, 1)) # [3,224,224]
        mean = [0.485, 0.56, 0.06]
        std = [0.229, 0.224, 0.225]
        image = image.astype(np.float32) / 255  # [0,1]
        image -= np.array(mean).reshape((3,1,1))
        image /= np.array(std).reshape((3,1,1))
        # image[0] -= mean # [-0.5, 0.5]
        # image /=std # []
        return image


    def __getitem__(self, index: int):
        image_size = 224
        name:str = self.images_list[index]
        image_name = name
        image = np.array(Image.open(image_name)) # uint8 [0-255]
        image = cv2.resize(image, (image_size,image_size))
        label_str = args.classes.index(name.split("\\")[-2])
        label = int(label_str)
        result = {
            "image": self.normalize(image),
            "label": label
        }
        return result
```

数据增强和数据读取



### 分类推理部分

detect.py 

```python
import argparse

import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.mobilenet import mobilenet_v2
import time
device = torch.device('cuda')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                                 std=std)
])

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = mobilenet_v2(pretrained=True) #     backbone + neck + head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(1280, len(args.classes)) # [bs, 1280] -> [bs, classes]

    def forward(self, x): # [bs,3,224,224]
        x = self.net.features(x) # [bs, 1280, 7, 7]  224//32
        x = self.avg_pool(x) # [bs, 1280, 1, 1]
        x = x.view(x.size(0), -1) # [bs, 1280]
        # x = torch.reshape()
        x = self.logit(x)
        return x
def predict():
    # net = torch.load('./ckpt/model.pth')
    # net = net.cuda()
    net = Net()
    net.load_state_dict(torch.load(args.model))
    net = net.cuda()
    net.eval()
    # net.to("cuda")
    # net.to(torch.device("cuda:0"))
    torch.no_grad()
    return net


def run(img):
    img = Image.fromarray(img[:, :, ::-1])
    # img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    outputs = torch.softmax(outputs, dim=1)
    score, predicted = torch.max(outputs, 1)
    return score[0].item(), predicted[0].item()

def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1 -2, y1-2), (x2+4, y2+4), (0, 0, 255), thickness=2)
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, default=['pause', 'hand'], help='your label')
    parser.add_argument('--source', type=int, default=0, help='your label')
    parser.add_argument('--model', type=str, default='./ckpt/model.pth', help='your label')
    parser.add_argument('--threshold', type=str, default='0.9', help='your label')
    args = parser.parse_args()
    net = predict()
    video = cv2.VideoCapture(args.source)
    while True:
        time1 = time.time()
        ret, img = video.read()
        img_copy = img
        roi = get_roi(img, 100, 324, 100, 324)
        # cv2.rectangle(img_copy, (95, 95), (328, 328), (0, 0, 255), thickness=1)
        if ret:
            cv2.imshow('img', roi)
            score, name = run(roi)
            name = args.classes[name]
            if float(score) >= float(args.threshold):
                cv2.putText(img_copy, str(name + ' '+str(round(score, 2))), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.imshow('frame', img_copy)
        time2 = time.time()
        print("Inference Time:", round(time2 - time1, 3))
        cv2.waitKey(5)
```

 效果图：

![1608470893189](Auto_maker_readme\12.png)

## 模型导出部分

mobilenet-v2 模型导出onnx部分

```python
import torch,onnx,collections
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = mobilenet_v2(pretrained=True) #     backbone + neck + head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(1280, num_classes) # [bs, 1280] -> [bs, classes]

    def forward(self, x): # [bs,3,224,224]
        x = self.net.features(x) # [bs, 1280, 7, 7]  224//32
        x = self.avg_pool(x) # [bs, 1280, 1, 1]
        x = x.view(x.size(0), -1) # [bs, 1280]
        # x = torch.reshape()
        x = self.logit(x)
        return x


print('notice !!!! ----> use python3 run this script!!! \n')
INPUT_DICT = 'ckpt\model.pth'
OUT_ONNX = 'ckpt\cls_model.onnx'

x = torch.randn(1, 3, 224, 224)
input_names = ["input"]
out_names = ["output"]
net = Net()
xmodel= torch.load(INPUT_DICT, map_location=torch.device('cuda'))
net.load_state_dict(xmodel)
net.eval()

torch.onnx.export(net, x, OUT_ONNX, export_params=True, training=False, input_names=input_names, output_names=out_names)
print('please run: python3 -m onnxsim test.onnx  test_sim.onnx\n')
print('convert done!\n')

```

yolov3 和yolov5 导出和openvino推理代码见：

yolov5 使用TensorRT推理代码见：





## 后续优化



这是cv调包侠的原创项目，没有参考过任何人，当然后来也看过网上用类似的方法实现了，但是我的更完整，完善，可移植性高，并且自带数据增强。

数据增强中，massic我没有使用，因为大多数模型自带massic，如果经过两次massic会更小，误检上会有问题。



### 优化tips1： 使用更多数据增强

但是不是越多增强越好~，比如我们没有使用翻转，因为模型自带翻转，以及hsv通道的增强，旋转和亮度，我们无需管，还有其他的数据正确策略，比如cut-mix ，等等，欢迎fork我的项目，并且完善~让项目更加简单，更高效：

github：https://github.com/CVUsers/Auto_maker



### 优化tips2：使用高质量相机采集，或者修改图片size获取更高清图片



### 优化tips3：使用更高质量跟踪算法：比如deepsort ，我已经做了，后续慢慢会开源

### 优化tips4：在使用的使用，尽量使用左右上下平移，这样会保证boxes更拟合。然后在调整了前后距离（大小）后，重新描框。

### 优化tips5：使用更接近场景的mix_up 图片。

### 优化tips6：更换更多场景，更多人物摄制，并获取更多数据。

### 优化tips7：大家一起加入进来一起完善！我有个优质公众号和两个深度学习交流群~大家进来一起交流，获取大量AI 深度学习数据集，和交流更优质的算法







## 总结

项目已发布：github：https://github.com/CVUsers/Auto_maker

**附带4w张玩手机数据：关注公众号回复**：**玩手机**

**公众号：DeepAi 视界**

二维码：
![1608470893189](Auto_maker_readme\13.jpg)




我们还可以标注什么数据? 

**答：绝大多数voc，coco数据，例如：猫狗，人，车，各种物体，但是过小的物体慎用~**



**作者 ：cv调包侠  本科大三 深度学习算法攻城狮实习僧 上海第二工业大学**