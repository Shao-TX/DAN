import os
from tqdm import tqdm
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from networks.dan import DAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=r"checkpoints\LIRIS\Batch_256\LR-0.0001\Epoch-6_Acc-0.9867.pth", help='Checkpoint name.')
    parser.add_argument('--num_class', type=int, default=6, help='Number of class.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')

    return parser.parse_args()

def plot_metric(y_true, y_pred):
    classes = ['Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    print("Report : \n", classification_report(y_true, y_pred, target_names=classes))

    mat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(mat, index = [i for i in classes], columns = [i for i in classes])

    sns.heatmap(df_cm, annot=True, cbar=False, square=True, fmt='.1f')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def class8_test(net, testLoader, device, model_class):
    Affect2Liris = {0:3, 1:2, 2:4, 3:5, 4:1, 5:0, 6:6, 7:7}
    net.eval()
    accuracy = 0
    count = 0

    y_true = []
    y_pred = []
    for imgs, label in tqdm(testLoader):
        imgs = imgs.to(device)
        label = label.to(device)
        output, feats, heads = net(imgs)
        _, predicted = torch.max(output, 1)

        if(model_class == 8):
            predicted_list = predicted.cpu().numpy().tolist()
            for id, i in enumerate(predicted_list):
                predicted_list[id] = Affect2Liris[i]
            new_predicted = torch.Tensor(predicted_list).to(device)

        if(int(new_predicted.item()) != 6 and int(new_predicted.item() != 7)):
            count += len(imgs)
            accuracy += (new_predicted == label).sum().item()

            y_true.append(int(label.item()))
            y_pred.append(int(new_predicted.item()))
            
    print("Test Accuracy: {} %".format(round(accuracy / count * 100, 3)))
    plot_metric(y_true, y_pred)

    return (accuracy / count)

def class6_test(net, testLoader, device, model_class):
    net.eval()
    accuracy = 0
    count = 0

    y_true = []
    y_pred = []

    for imgs, label in tqdm(testLoader):
        imgs = imgs.to(device)
        label = label.to(device)
        output, feats, heads = net(imgs)
        _, predicted = torch.max(output, 1)

        count += len(imgs)
        accuracy += (predicted == label).sum().item()

        y_true.append(int(label.item()))
        y_pred.append(int(predicted.item()))

    print("Test Accuracy: {} %".format(round(accuracy / count * 100, 3)))
    plot_metric(y_true, y_pred)

    return (accuracy / count)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DAN(num_class=args.num_class, num_head=4)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model = model.to(device)
    # print(model)
    
    data_transforms_val = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])      

    test_dataset = datasets.ImageFolder(r'datasets\LIRIS\split_cut_myself_image811\test', transform = data_transforms_val)    # loading statically
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers =1, shuffle = False, pin_memory = True)    
    
    if(args.num_class == 8):
        class8_test(model, test_loader, device, args.num_class)
    else:
        class6_test(model, test_loader, device, args.num_class)