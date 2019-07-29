"""Common utilities"""
import os

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils import data

def get_data_transforms():
    """Return transforms for loading images to be used by the deep network."""
    return {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }


def load_datasets(data_dir):
    """Returns both datasets and loaders for images in the given folder.
    
    It assummes the data_dir has three folders: 'train', 'test', and 'valid'.
    Each of these folders should organize images as required by torchvision,
    where each label should have its own folder with all the samples for it.
    """
    data_transforms = get_data_transforms()
    image_datasets = {
        k: datasets.ImageFolder(os.path.join(data_dir, k), 
                                transform=data_transforms[k])
        for k in ['train', 'test', 'valid']
    }
    dataloaders = {
        k: data.DataLoader(image_datasets[k], batch_size=64, 
                           shuffle=(k == 'train'))
        for k in ['train', 'test', 'valid']
    }
    return image_datasets, dataloaders


def get_pre_trained_model(arch):
    """Returns a pre-trained torchvision model ready for transfer learning."""
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_classifier(input_units, hidden_units, output_units, dropout):
    """Get a neural network classifier with a single hidden layer."""
    return nn.Sequential(nn.Linear(input_units, hidden_units),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(hidden_units, output_units),
                         nn.LogSoftmax(dim=1))


def save_checkpoint(filepath, arch, model, hyperparams):
    """Save a model checkpoint.pth at the given path."""
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hyperparams': hyperparams,
    }
    torch.save(checkpoint, filepath)

    
def load_checkpoint(filepath):
    """Loads a checkpoint and returns the model obtained from it."""
    ch = torch.load(filepath)
    model = getattr(models, ch['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = ch['class_to_idx']
    hparams = ch['hyperparams']
    model.classifier = nn.Sequential(
        nn.Linear(hparams['input_units'], hparams['hidden_units']),
        nn.ReLU(),
        nn.Dropout(hparams['dropout']),
        nn.Linear(hparams['hidden_units'], hparams['output_units']),
        nn.LogSoftmax(dim=1))
    model.load_state_dict(ch['state_dict'])
    return model 


def process_image(image_tensor):
    """Scales, crops, and normalizes a PIL image represented as a tensor."""
    pil_img = Image.fromarray(image_tensor.numpy())
    
    # Resize image to a size of 256, but respecting aspect ratio.
    # Followed default torchvision.transforms.Resize behaviour
    w, h = pil_img.size
    if h > w:
        scale_size = (256, int(256 * h / w))
    else:
        scale_size = (int(256 * w / h), 256)
    pil_img = pil_img.resize(scale_size, Image.BILINEAR)
    
    # Crop a center of 224x224 
    crop_size = 224
    w, h = pil_img.size
    w_cut = (w - crop_size) / 2.0
    h_cut = (h - crop_size) / 2.0
    pil_img = pil_img.crop((w_cut, h_cut, w - w_cut, h - h_cut))
    
    # Get numpy array so we can normalize the values 
    np_image = np.array(pil_img)
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Return a tensor, taking into account that torchvision
    # presents the RGB dimension first. Also make sure
    # a torch.FloatTensor is returned as that's the type
    # the model expects
    img_tensor = torch.tensor(np_image.transpose((2, 0, 1)))
    return img_tensor.type(torch.FloatTensor)


def imshow(image_tensor, ax=None, title=None):
    """Displays an image from a torchvision's tensor"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image_tensor.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 
    # or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title is not None:
        ax.set_title(title)
    ax.imshow(image)
    
    return ax