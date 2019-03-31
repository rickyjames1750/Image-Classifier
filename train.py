import argparse
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms, utils, models

### $
def main():

    print('Generating validation, training, and experimenting databases')

   
    data_dir = in_arg.database
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transform
    #for training augmentation
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.486, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    img_train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    img_test_datasets = datasets.ImageFolder(test_dir, transform =data_transforms)
    img_valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)# Data Loading #TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(img_train_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(img_test_datasets, batch_size = 32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(img_valid_datasets, batch_size = 32, shuffle = True)


    def view_batch(title, bunch):
        plt.figure(figsize=(12, 12)) 
        grid = utils.make_grid(bunch[0], normalize=True) 
        plt.imshow(grid.numpy().transpose((1, 2, 0))) 
        plt.title(title) 
    pass

    bunch = next(iter(testloader))
    view_batch('Batch to Testloader', bunch)

    print('Finshed and Creating model')

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    #--arch can also be used as a different architectures available from torchvision.models
    if in_arg.vgg16 == True:
        model = models.vgg16(pretrained = True)
        input_size = model.classifier[0].in_features
        print('VGG16 is CNN base model')
    elif in_arg.dnet == True:
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
        print('Densenet121 is CNN base model')
    else:
        model = models.vgg16(pretrained = True)
        input_size = model.classifier[0].in_features
        print('Specification Error,VGG16 CNN base model is having issues')

    output_size = 102 
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 8000)), ('dp1', nn.Dropout(0.6)), ('relu', nn.ReLU()), ('fc2', nn.Linear(8000, 2000)), ('dp2', nn.Dropout(0.6)), ('relu2', nn.ReLU()), ('fc3', nn.Linear(2000, output_size)), ('output', nn.LogSoftmax(dim = 1))
    ]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.classifier = classifier
    model.to(device)
    print(classifier)
    model.classifier = classifier
    model.to(device)
    print('Finished - beginning training')

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr = in_arg.lr)

    ###
    model = train(model, trainloaders, testloaders, validationloaders, criterion, optimizer)

    print('Training Done')
    print('Saving model .pth file')

    torch.save(model.classifier.state_dict(), 'checkpoint.pth')
    print(model)
    print('Done' + 'End of program')#### Building and training the classifier

    ###
    
    running_loss = 0
    epochs = 1
    steps = 0
    print_every = 80

    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
            
                with torch.no_grad():
                    right, final = validation(model, validationloader)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format((final - right)/final),
                      "Test Accuracy: {:.3f}".format(right/final))
            
                running_loss = 0
            
                model.train()
    model.eval()
### $
def validation(model, loader):
    right = 0
    final = 0
    for inputs, labels, in loader:
        #Feedforward
        inputs, labels = inputs.to(device), labels.to(device)
        final += len(labels)
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)

        right += (labels == predicted).sum().item()
        
    return right, final
### $
def get_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type=str, default='./flowers/test/1/image_06760.jpg', 
    help='Path twoards image')

    parser.add_argument('--model', type=str, default='checkpoint.pth', 
    help='checkpoint activates NN')

    parser.add_argument('--k', type=int, default=1, 
    help='Total relevancy predictions for the user')

    parser.add_argument('--cuda', action='store_true' ,
    help='Runs the programming to the GPU')

    parser.add_argument('--dict', type=str, default=None, 
    help='decides names of the outputs')
    
    parser.add_argument('--vgg16', action='store_true',
    help='CNN -> VGG16')

    parser.add_argument('--dnet', action='store_true',
    help='CNN -> DenseNet121')

    return parser.parse_args()