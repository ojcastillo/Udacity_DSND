"""Train a new deep network on a data set using transfer learning.

Usage:
  train.py DATA_DIRECTORY   [-h | --help]   [--save_dir=<dir_path>]
                            [--arch=<arch>] [--learning_rate=<rate>]
                            [--input_units=<units>] [--hidden_units=<units>]
                            [--epochs=<epochs>] [--dropout=<probability>] [--gpu]

Arguments:
  DATA_DIRECTORY            Path to folder with images

Options:
  -h --help                 Show this help message.
  --save_dir=<dir_path>     Where to save the model checkpoint [default: ./]
  --arch=<arch>             Model for learning transfer [default: densenet121]
  --learning_rate=<rate>    Learning rate for optimizer [default: 0.001]
  --input_units=<units>     Force the number of input units
  --hidden_units=<units>    Number of units in hidden layer [default: 256]
  --dropout=<probability>   Dropout probability of hidden layer [default: 0.33]
  --epochs=<epochs>         Number of training epochs [default: 3]
  --gpu                     Use GPU for model computation if set
"""
import logging
import os
import time

from docopt import docopt
import torch
from torch import nn, optim

import common
from workspace_utils import keep_awake

def train_nn_model(nn_model, optimizer, trainloader, validloader, 
                   device, epochs, report_every=20):   
    """Run training loop."""
    criterion = nn.NLLLoss()
    nn_model.to(device);
    running_loss = 0.0
    start_time = time.time()
    for epoch in keep_awake(range(epochs)):
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = nn_model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % report_every == 0:
                val_loss = 0
                accuracy = 0
                nn_model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = nn_model(inputs)
                        val_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)  # inverse of log
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        device_type = torch.FloatTensor
                        if device == torch.device('cuda'):
                            device_type = torch.cuda.FloatTensor
                        accuracy += torch.mean(equals.type(device_type)).item()

                val_size = len(validloader)
                logging.info(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Steps: {steps}.. "
                    f"Time: {(time.time() - start_time):.3f}s.. "
                    f"Running loss: {running_loss/report_every:.3f}.. "
                    f"Validation loss: {val_loss/val_size:.3f}.. "
                    f"Validation accuracy: {100 * accuracy/val_size:.3f}%"
                )
                running_loss = 0
                nn_model.train()
    logging.info(f"Total training time: {(time.time() - start_time):.3f}s")

def test_model(nn_model, testloader, device):
    """Run model over the test set"""
    criterion = nn.NLLLoss()
    nn_model.to(device);
    test_loss = 0
    accuracy = 0
    nn_model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = nn_model(inputs)
            test_loss += criterion(logps, labels).item()

            ps = torch.exp(logps)  # inverse of log
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            device_type = torch.FloatTensor
            if device == torch.device('cuda'):
                device_type = torch.cuda.FloatTensor
            accuracy += torch.mean(equals.type(device_type)).item()
    nn_model.train()
    logging.info(f"Test loss: {test_loss/len(testloader):.3f}.. "
                 f"Test accuracy: {100 * accuracy/len(testloader):.3f}%")

def get_classifier_input_size(nn_model):
    """Tries to guess the number of input units required by the classifier."""
    num_ftrs = None
    try:
        num_ftrs = nn_model.classifier[0].in_features
    except:
        try:
            num_ftrs = nn_model.classifier.in_features
        except:
            logging.exception('Failed to guess the number of input units. Please provide it through --input_units.')
    return num_ftrs
    
def main(args):
    """Starting point"""
    logging.basicConfig(format='%(asctime)s -- %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    model_arch = args['--arch']
    logging.info('Setting up %s model...', model_arch)
    image_datasets, dataloaders = common.load_datasets(args['DATA_DIRECTORY'])
    device = torch.device('cuda' if args['--gpu'] else 'cpu')
    
    hidden_units = int(args['--hidden_units'])
    dropout = float(args['--dropout'])
    epochs = int(args['--epochs'])
    learning_rate = float(args['--learning_rate'])

    nn_model = common.get_pre_trained_model(model_arch)
    logging.info('Original pre-trained classifier: %s', nn_model.classifier)
    num_ftrs = get_classifier_input_size(nn_model) if args['--input_units'] is None else int(args['--input_units'])
    num_classes = len(image_datasets['train'].classes)
    nn_model.classifier = common.get_classifier(num_ftrs, hidden_units,
                                                num_classes, dropout)
    nn_model.class_to_idx = image_datasets['train'].class_to_idx
    
    logging.info(f'Running training loop...')
    optimizer = optim.Adam(nn_model.classifier.parameters(), lr=learning_rate)
    train_nn_model(nn_model, optimizer, dataloaders['train'], 
                   dataloaders['valid'], device, epochs)
    
    checkpoint = os.path.join(args['--save_dir'], 'checkpoint.pth')
    logging.info(f'Saving checkpoint in "%s"...', checkpoint)
    common.save_checkpoint(checkpoint, model_arch, nn_model, {
        'input_units': num_ftrs,
        'hidden_units': hidden_units,
        'output_units': num_classes,
        'dropout': dropout,
    })
    
    logging.info(f'Running model over test set...')
    test_model(nn_model, dataloaders['test'], device)
    
    logging.info('My job is done, going gently into that good night!')
    return 0

if __name__ == '__main__':
    args = docopt(__doc__)
    exit(main(args))