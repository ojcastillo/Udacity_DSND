"""Predict a flower name from an image.

Usage:
  predict.py IMAGE_PATH  CHECKPOINT   [-h | --help] [--top_k=<K>] [--gpu]
                                      [--category_names=<json_file>]

Arguments:
  IMAGE_PATH                Path to the image we want to predict
  CHECKPOINT                Path to model checkpoint file to restore model from

Options:
  -h --help                     Show this help message.
  --top_k=<K>                   Number of top most likely classes [default: 5]
  --gpu                         Use GPU for model computation if set
  --category_names=<json_file>  Path to json file with category mapping
"""
from docopt import docopt
import logging
import json

import numpy as np
from PIL import Image
import torch
from torch import nn

import common

# Implement the code to predict the class from an image file
def predict(img_path, model, device, topk=5):
    """Predict classes of an image using a trained deep learning model."""
    img = Image.open(img_path)
    img_tensor = common.process_image(torch.tensor(np.array(img)))
    img_tensor.unsqueeze_(0)  # so it looks like a batch of 1 image 
    
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model(img_tensor)
        ps = torch.exp(logps)  # inverse of log
        top_ps, top_classes = ps.topk(topk, dim=1)
    model.train()
    return top_ps[0], top_classes[0]


def main(args):
    """Starting point"""
    logging.basicConfig(format='%(asctime)s -- %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    img_path = args['IMAGE_PATH']
    checkpoint = args['CHECKPOINT']
    topk = int(args['--top_k'])
    device = torch.device('cuda' if args['--gpu'] else 'cpu')
    
    category_names = None
    if args['--category_names']:
        with open(args['--category_names'], 'r') as f:
            category_names = json.load(f)
    
    logging.info('Rebuilding model from checkpoint "%s"...', checkpoint)
    model = common.load_checkpoint(checkpoint)
    
    logging.info('Predicting top %s class(es) for "%s"', topk, img_path) 
    probs, classes = predict(img_path, model, device, topk)
    
    probs = [p.item() for p in probs]
    classes = [cl.item() for cl in classes]
    if category_names is not None:
        idx_to_class = {v:k for k,v in model.class_to_idx.items()}
        classes = [category_names[idx_to_class[cl]] 
                   for cl in classes]
    logging.info('Predicted classes: %s', classes)
    logging.info('Predicted class probabilities: %s', probs)
                 
    logging.info('My job is done, going gently into that good night!')
    return 0

if __name__ == '__main__':
    args = docopt(__doc__)
    exit(main(args))
