import torch
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler
from torchvision import transforms, datasets, models
import torch.nn as nn

# Data science tools
import numpy as np

# Visualizations
import matplotlib.pyplot as plt

# Image manipulations
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import copy
from timeit import default_timer as timer


def get_resnet(num_classes):
    model = models.resnet101(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))#, nn.LogSoftmax(dim=1))
    # model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    return model

def get_densenet(num_classes):
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(dim=1))
    model_ft.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))
    return model

def get_dataloaders(data_dir, valid_size, batch_size, image_transforms, show_sample=False):
    np.random.seed(12)
    torch.manual_seed(12)
    data = {
        'train':
        datasets.ImageFolder(data_dir, image_transforms['train']),
        'val':
        datasets.ImageFolder(data_dir, image_transforms['val']),
    }
    train_idx, valid_idx = [], []
    counts = (data['train'].targets.count(i) for i in data['train'].class_to_idx.values())
    acc = 0
    for numb in counts:
        valid_split = int(np.floor(valid_size * numb))
        indices = list(range(acc, acc+numb))
        acc += numb
        np.random.shuffle(indices)
        train_idx.extend(indices[:numb-valid_split])
        valid_idx.extend(indices[numb-valid_split:])

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    show_transform=False
    if show_transform:
        ex_path = os.path.join(data_dir, os.listdir(data_dir)[0])
        ex_path = os.path.join(ex_path, os.listdir(ex_path)[1500])
        ex_img = Image.open(ex_path)
        imshow(ex_img)

        t = image_transforms['train']
        plt.figure(figsize=(24, 24))

        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            _ = imshow_tensor(t(ex_img), ax=ax)

        plt.tight_layout()
        plt.show()
        # exit()

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(data['train'],  batch_size=9, sampler=train_sampler,)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        plot_images(images, labels, data['train'].classes)
        # exit()

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'],  batch_size=batch_size, sampler=train_sampler, drop_last=True),
        'val': DataLoader(data['val'],  batch_size=batch_size-6, sampler=valid_sampler, drop_last=True),
    }
    return dataloaders


def plot_training_stats(history: dict):
    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
            history['loss'][c], label='loss '+c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Losses')
    # plt.show()
    plt.savefig("resnetloss.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
             history['acc'][c], label='acc ' + c)#100 *
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    # plt.show()
    plt.savefig("resnetacc.png")
    plt.close()
            



# Auxilary ----------------------------------------------------


def plot_images(images, cls_true, label_names, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    X = images.numpy().transpose([0, 2, 3, 1])
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(np.clip(X[i, :, :, :], 0, 1), interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

