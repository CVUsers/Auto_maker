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
