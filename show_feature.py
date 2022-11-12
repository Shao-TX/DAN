#%%
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import os
import argparse

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from networks.dan import DAN

#%%
# 提取不同層輸出的 主要代碼
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='datasets/LIRIS/cut_image/valid/disgust/1.jpg', help='Image path.')
    parser.add_argument('--checkpoint', type=str, default="liris_epoch10_batch256_acc0.9636", help='Checkpoint name.')
    parser.add_argument('--layer', type=int, default=0, help='Layer of model.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    num_class = 5 if 'liris' in args.checkpoint else 8

    img_path = args.img_path
    checkpoint_path = "./checkpoints/" + args.checkpoint + ".pth"
    model_layer = args.layer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DAN(num_head=4, num_class=num_class, pretrained=False)
    checkpoint = torch.load(checkpoint_path ,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(device)
    model.eval()

    summary(model, (3, 224, 224))

    data_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]) 


    print(model.features)

    #%%
    img = cv2.imread(img_path)
    img = Image.fromarray(img) # np.array to PIL Image
    img = data_transforms(img)
    img = img.view(1,3,224,224)

    conv_out = LayerActivations(model.features, model_layer)  # Model 第 model_layer 層的輸出
    o = model(img.cuda())
    conv_out.remove()
    act = conv_out.features  # act : 第 model_layer 層輸出的特徵

    output_channel = len(act[0]) # act[0] 中包含 Model 第 model_layer 層輸出的通道數 => [-1, 64, 112, 112] => 通道 64 個 
    print("Number output_channel : ", output_channel)

    #%%
    # 可視化輸出
    fig = plt.figure(figsize=(12, 12))

    # fig_column, fig_row =  int(output_channel/4), 4

    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[]) # 列、行、位置
        ax.imshow(act[0][i].detach().numpy(), cmap="gray")

    plt.show()
# %%
