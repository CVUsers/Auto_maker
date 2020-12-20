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
