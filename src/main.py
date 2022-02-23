import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batchsize", default=2, type=int)

######
parser.add_argument("--num_classes", default=3, type=int)
parser.add_argument("--using_unet", default=0, type=int)
parser.add_argument("--train_all", default=1, type=int)
######

args = parser.parse_args()

######
num_classes = args.num_classes
using_unet = True if args.using_unet==1 else False
train_all = True if args.train_all==1 else False

other_than_five_classes = True if num_classes != 5 else False
######



bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize

# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3(using_unet=using_unet, train_all=train_all)
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score, 'accuracy' : accuracy_score}


print("Begin training process with the following properties:")
print(f'Number of epochs = {epochs}\nBatchsize = {batchsize}\nNumber of classes = {num_classes}\nUsing unet: {using_unet}\nTrain the whole network: {train_all}')

# Create the dataloader
dataloaders = datahandler.get_dataloader_sep_folder(
    data_dir, batch_size=batchsize, other_than_5_classes=other_than_five_classes, num_classes=num_classes, with_aug=True)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs, using_unet=using_unet)


# Save the trained model
torch.save(model.state_dict(), os.path.join(bpath, 'weights.pt'))
