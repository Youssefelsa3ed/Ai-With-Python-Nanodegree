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

def data_transformation(data_dir='ImageClassifier/flowers'):
    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    train_data = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    validate_data = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)
    
    return train_data, validate_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_method" , type = str , default ="cpu" ,help = "Training method gpu or cpu")
    parser.add_argument("--Arch",type = str , default = 'vgg16' , help ="Either vgg16 or densenet121")
    parser.add_argument("--learn_rate" , type = float , default = 0.001 , help ="Learning Rate")
    parser.add_argument("--hidden_units" , type = int , default = 4096 , help = "Number of hidden units")
    parser.add_argument("--num_epochs" , type= int, default = 3 ,help ="Training epochs")
    return parser.parse_args()

def create_network(arch, learn_rate, hidden_layer, dropout = 0.5):
    if(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif(arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        in_features = 1024   
    else:
        print("Architecture must be either VGG16 or DenseNet121")
        
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

def train_model(model, trainloader, validateloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
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
                    test_loss, accuracy = calculate_accuracy(model, validateloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validateloader)),
                      "Accuracy: {:.3f}".format(accuracy/len(validateloader)))
                
                running_loss = 0
                model.train()
    print("Training is done!")
    

def calculate_accuracy(model, validateloader, criterion, device):
    accuracy = 0
    test_loss = 0
    for images, labels in validateloader:
        images, labels = images.to(device) , labels.to(device)
        model.to(device)
            
        output = model.forward(images)
        test_loss += criterion(output, labels)

        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
    
    
def save_chk(model, in_args):
    in_features = 25088
    if in_args.Arch == 'densenet121':
        in_features = 1024    
    torch.save({'arch':in_args.Arch,
                'state_dict': model.state_dict(),
                'hidden_units' : in_args.hidden_units ,
                'class_to_idx': model.class_to_idx,
                'in_features':in_features,
                'epochs': in_args.num_epochs,
                'learning_rate':in_args.learn_rate} , 'checkpoint.pth')

    
def main():
    in_args = get_args()
    
    train_data, validate_data = data_transformation()
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size=32)
    
    model, optimizer, criterion = create_network(in_args.Arch, in_args.learn_rate, in_args.hidden_units)
    model.class_to_idx =  train_data.class_to_idx
    
    device = 'cpu'
    if(in_args.train_method == 'gpu' and torch.cuda.is_available()):
        device = 'cuda'
    train_model(model, trainloader, validateloader, in_args.num_epochs, 40, criterion, optimizer, device)
    save_chk(model, in_args)
    
    
if __name__=='__main__':
    main()
    