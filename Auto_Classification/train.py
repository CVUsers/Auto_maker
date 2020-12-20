"""
基于mobilenet v2的分类
"""
import argparse

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import albumentations as A
from PIL import Image
import cv2
import glob,os
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50
from tqdm import tqdm
import detect
from torch.utils.data import DataLoader, Dataset


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


def train_one(dataloader, model, optimizer, loss_fn, schduler=None):
    model.train()
    # model.eval()

    optimizer.zero_grad()

    tk = tqdm(dataloader)
    for bi, data in enumerate(tk):
        images = data['image'].cuda() # [bs,3,224,224]   [bs,time_step,fv]  [b,25,H,W]
        labels = data['label'].cuda()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if schduler is not None:
            schduler.step()

        tk.set_postfix(loss=loss.item())


def val_one(dataloader, model, loss_fn):
    model.eval()
    losses = 0.
    count = 0
    acc = []
    with torch.no_grad():
        for bi, data in enumerate(dataloader):
            images = data['image'].cuda()  # [bs,3,224,224]   [bs,time_step,fv]  [b,25,H,W]
            labels = data['label'].cuda() # [] bs
            # pre = detect.run(images.item()[0])
            # if pre == labels.item():
            #     pass
            outputs = model(images)
            output_label = torch.softmax(outputs, dim=1) #[bs, num_classes]
            pre_labels = output_label.argmax(dim=1) # [bs]
            accuracy = (pre_labels == labels).tolist()
            acc.extend(accuracy)
            # print("**", labels)
            loss = loss_fn(outputs, labels) # float
            losses += loss.item()
            count += images.size(0)

    scores = {
        "loss": losses / count,
        "accuracy": np.array(acc).mean()
    }
    for k, v in scores.items():
        print(k, v)
    return scores


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


def prepare_data():
    images_list = glob.glob("./data/train/*/*")
    val_list = glob.glob("./data/test/*/*")
    return images_list, val_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, default=['pause', 'hand'], help='your label')
    parser.add_argument('--dtype', type=str, default='pause', help='your label')
    parser.add_argument('--epochs', type=int, default=100, help='your label')
    parser.add_argument('--batch_size', type=int, default=64, help='your label')
    args = parser.parse_args()
    os.makedirs("ckpt", exist_ok=True)
    images_list, val_list = prepare_data()
    run(images_list, val_list)











