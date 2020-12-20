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