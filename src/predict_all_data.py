import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import model
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import accuracy_score


def return_nearest_cluster(point, clusters):
    dists = np.abs(clusters - point)
    return clusters[np.argmin(dists)].item()


def quantization(mask, num_classes):
    out = np.zeros(shape=(400, 400), dtype=float)
    flat_mask = mask.reshape(1, -1).T
    clusters = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    for i in range(400):
        for j in range(400):
            out[i][j] = return_nearest_cluster(mask[i][j], clusters)
    return out


def indexed_cluster_map(mask, clusters):
    out = np.zeros(shape=mask.shape,  dtype=np.uint8)
    height, width = mask.shape[0], mask.shape[1]
    for h in range(height):
        for w in range(width):
            dists = np.abs(clusters - mask[h][w])
            out[h][w] = np.argmin(dists)
    return out


def accuracy_per_sample(quantized_output, true_mask, num_classes):
    flat_mask = true_mask.reshape(1, -1).T
    kmeans = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    clusters = sorted([kmeans[i].item() for i in range(len(kmeans))])
    index_mask = indexed_cluster_map(true_mask, clusters)
    index_output = indexed_cluster_map(quantized_output, clusters)
    return accuracy_score(index_mask.flatten(), index_output.flatten())


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def quantization_fix_thresholds(img, thresholds=(0.5, 0.7, 0.9, 1)):
    out = np.zeros(shape=(400, 400), dtype=float)
    for i in range(400):
        for j in range(400):
            coord = img[i][j]
            if coord <= thresholds[0]:
                out[i][j] = 0
            elif thresholds[0] < coord <= thresholds[1]:
                out[i][j] = 0.3
            elif thresholds[1] < coord <= thresholds[2]:
                out[i][j] = 0.6
            else:
                out[i][j] = 1
    return out


def img_to_mask(img_pt, num_classes):
    img = cv2.imread(img_pt)
    img = cv2.resize(img, (400, 400)).transpose(2, 0, 1).reshape(1, 3, 400, 400)
    with torch.no_grad():
        mask = model(torch.from_numpy(img).type(torch.FloatTensor) / 255)
        quantized_mask = quantization(mask, num_classes)
    return img, mask, quantized_mask


def create_dir_masks(in_pt, num_classes):
    import os
    imgs_and_masks = []
    for root, dirs, files in os.walk(in_pt):
        for name in files:
            img_pt = os.path.join(root, name)
            mask = img_to_mask(img_pt, num_classes)
            imgs_and_masks.append(mask)
    return imgs_and_masks


def show_many_imgs(x, labels_list, amount, out_path=''):
    """
    Doesn't work in Nova
    Args:
        x:
        labels_list:
        amount:
        out_path:

    Returns:

    """
    f, axarr = plt.subplots(amount, amount)  # here it got stuck
    plot_ndxs = [(i, j) for i in range(amount) for j in range(amount)]
    for i in range(amount * amount):
        e = axarr[plot_ndxs[i][0], plot_ndxs[i][1]].imshow(x[i])
        f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)
        # if len(labels_list) != 0:
        axarr[plot_ndxs[i][0], plot_ndxs[i][1]].set_title(labels_list[i])
    if len(out_path) > 0:
        plt.savefig(out_path)
    else:
        plt.show()


def create_masks(data_dir, num_classes, weights_filename, using_unet=False):
    results = []

    import datahandler, model

    ####
    other_than_five_classes = True if num_classes != 5 else False
    ####

    dataloaders = datahandler.get_dataloader_sep_folder(
        data_dir, batch_size=1, other_than_5_classes=other_than_five_classes, num_classes=num_classes, with_aug=True)
    k = 0
    train_acc, test_acc = 0, 0

    model = model.createDeepLabv3(using_unet=using_unet)
    # Load the trained model
    weights_filepath = "./weights/" + weights_filename + ".pt"
    model.load_state_dict(torch.load(weights_filepath, map_location=torch.device('cpu')))

    for phase in ['Train', 'Test']:
        i = 0
        model.eval()  # Set model to evaluate mode
        # Iterate over data.
        for sample in tqdm(iter(dataloaders[phase])):
            i += 1

            inputs = sample['image']
            mask = sample['mask']
            outputs = model(inputs)
            if using_unet:
                outputs = {'out': outputs}
            output = outputs['out'].cpu().detach().numpy()[0][0]
            quantized_mask = quantization(output, num_classes)

            #####
            if phase == 'Train':
                train_acc += accuracy_per_sample(quantized_mask, mask[0][0].numpy(), num_classes)
                print(train_acc)
            else:
                test_acc += accuracy_per_sample(quantized_mask, mask[0][0].numpy(), num_classes)
                print(test_acc)
            #####

            sample_results = [np.array(inputs[0]).transpose(1, 2, 0), mask[0][0], output, quantized_mask]
            results.append(sample_results)
            k += 1
        #####
        if phase == 'Train':
            train_acc = train_acc / i
        else:
            test_acc = test_acc / i
        #####
    print(f'For architechture {weights_filename}: Training accuracy={train_acc}, Test accuracy = {test_acc}')
    pickle.dump(results, open("seg_results.p", "wb"))


if __name__ == '__main__':
    create_masks('./data', 4, "4class_withacc_new_quan", using_unet=False)
