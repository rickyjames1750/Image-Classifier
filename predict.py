import argparse
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms, utils, models

###$ Predicting Classes 
#Displaying class names featured in this code segment for both main and load_checkpoint
def main():
    model = model_loads(in_arg.model)
    model.eval()
    image_path = in_arg.dir
    with torch.no_grad():
        probs, classes = predict(image_path=image_path, model=model,k=in_arg.k)
        real_names = [model.idx_to_class[x] for x in classes] 
        if in_arg.dict == None:
            label_id = [model.class_to_name[x].title() for x in real_names]
        else:
            with open(in_arg.dict, 'r') as f:
                cat_to_name = json.load(f)
            label_id = [cat_to_name[x].title() for x in real_names] #Displaying class names
        for i in range(len(probs)):
            print('probability is: {:.2f}%'.format(label_name[i], probs[i] * 100))
        print('Executed Program')
    return 
###$
def load_checkpoint(filepath):
    checkpoint = torch.load('checkpoint.pth')
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
    in_args = parser.parse_args()
    in_args.device = None
    
    model = checkpoint['pretrained_model']
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not in_args.disable_cuda and torch.cuda.is_available():
        in_args.device = torch.device('cuda')
    else:
        in_args.device = torch.device('cpu') 

    
    model.class_to_name = checkpoint['class_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = dict([(v, k) for k, v in model.class_to_idx.items()])
    
    #load_checkpoint('checkpoint.pth')
    #print(model)
    model.to(device)
    return model
nn.AdaptiveAvgPool2d(1)
###$
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

    return parser.parse_args()
###$
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #CODE BELOW
    # TODO: Process a PIL image for use in a PyTorch model
    process_trans = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    packaged_image = process_trans(image)
    packaged_image = np.asarray(packaged_image) 
    
    return packaged_image
###$ Prediciting with GPU and Top 5 k classes in code segment 
def predict(image_path, model, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.to(device)
    model.eval()
    
    pil_image = Image.open(image_path).convert('RGB')
    processed_img = process_image(pil_image)
    torch_image = torch.from_numpy(processed_img)
    torch_image = torch_image.unsqueeze_(0)
    torch_image = torch_image.float().to(device)  
    
    output = model(torch_image).exp_().topk(k) 

    classes = output[1].cpu().numpy()   
    probs = output[0].cpu().numpy()
    
    return np.reshape(probs, (k)), np.reshape(classes, (k))
