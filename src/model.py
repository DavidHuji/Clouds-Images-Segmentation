""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch


def createDeepLabv3(outputchannels=1, using_unet=False, train_all=True):
    if using_unet:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)

    if train_all == False:
        for param in model.parameters():
            param.requires_grad = False
    #print(model)
    if using_unet:
        FirstTrial = False
        if FirstTrial:  # ruing memory in CPU
            model.conv = DeepLabHead(32, outputchannels)
        else:
            model.conv = torch.nn.Sequential(
                            torch.nn.Conv2d(32, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.1, inplace=False),
                            torch.nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1)),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(21, 1, kernel_size=(1, 1), stride=(1, 1)),
                            torch.nn.Tanh())
    else:
        model.classifier = DeepLabHead(2048, outputchannels)

    #print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', model)
    # Set the model in training mode
    model.train()
    return model

if __name__ == '__main__':
    m = createDeepLabv3()

