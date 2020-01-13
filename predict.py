import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
import argparse

def data_transformation(data_dir='flowers'):
    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    
    test_data = datasets.ImageFolder(data_dir + '/test', transform=data_transforms)
    
    return test_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_method" , type =str, default = "cpu" , help="Training method cpu or gpu")
    parser.add_argument("--dir", type=str, default ="flowers/test/14/image_06052.jpg", help="Image path")
    parser.add_argument("--chk_pth" , type=str, default="checkpoint.pth", help="Checkpoint Path" )
    parser.add_argument("--top_k" ,type = int,default = 5 , help ="Top K classes")
    parser.add_argument("--map" , type=str, default ="cat_to_name.json" , help="Mapping name file path")
    return parser.parse_args()

def load_model(filepath):
    checkpoint = torch.load(filepath)
    model, criterion, optimizer = create_network(arch = checkpoint['arch'],learn_rate=checkpoint['learning_rate'], hidden_layer=checkpoint['hidden_units'], in_features=checkpoint['in_features'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
def create_network(arch, learn_rate, hidden_layer, in_features, dropout = 0.5):
    if(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))]))
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learn_rate)
    model.cuda()
        
    return model, optimizer, criterion

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)

    return transform(pil_image)
    
def predict(image_path, model, device, topk):
    model.eval()
    model.to(device)
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        if(device == 'cuda'):
            output = model.forward(img.cuda())
        else:
            output = model.forward(img)
    
    ps = torch.exp(output.data)
    return ps.topk(topk)    
    
def main():
    in_args = get_args()
    
    test_data = data_transformation()
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    device = 'cpu'
    if(in_args.train_method == 'gpu' and torch.cuda.is_available()):
        device = 'cuda'
    model = load_model(in_args.chk_pth)
    
    probabilities = predict(in_args.dir, model, device, in_args.top_k)
    probability = np.array(probabilities[0][0])
    names = np.array(probabilities[1][0])
    
    with open(in_args.map, 'r') as f:
        model_to_name = json.load(f)
    
    model_keys = {}
    for key, val in model.class_to_idx.items():
        model_keys[val] = key
    classes = [model_keys[item] for item in names]
    
    for i in range(in_args.top_k):
        print("{} with a probability of {:.2f}".format(model_to_name[classes[i]], probability[i]))
    
if __name__=='__main__':
    main()