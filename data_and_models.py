import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.datasets as datasets

def retrieve_indices(classes,dataset):
    
    '''
    retrieve_indices(classes,dataset)
    
    find indices in dataset that are of classes specified in classes
    arguments:
        classes (list): list of classes to look for in the dataset, as list of integers
        dataset (torch Dataset): a torch Dataset in which we search for (match targets to classes)
        
    returns: list of indices that belong to the desired classes
    ''' 
    
    
    class_indices=[i for i,label in enumerate(dataset.targets) if int(label) in classes] # label itself was a tensor :/
    return class_indices

def create_mnist_set(root,train,classes):
    
    '''
    create_mnist_set(root,train,classes)
    
    Create dataset from torchvision's MNIST data
    arguments:
        root (string): root directory to download files to/to look for files
        train (bool): whether work with pre-defined train- or testset
        classes (list): list of classes to include in the dataset, as list of integers
        
    returns the dataset defined by the above arguments
    '''
    
    mnist_trainset = datasets.MNIST(root=root, train=train, download=True, transform=lambda x: np.array(x))
    return Subset(mnist_trainset,retrieve_indices(classes,mnist_trainset))

class mnist_binary_cnn(nn.Module):

    def __init__(self,batch_transform=lambda x: (x/255.).reshape(-1,1,28,28), loss=nn.BCELoss()):
        super(mnist_binary_cnn,self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=4,stride=2)
        self.conv2=nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=20,out_channels=10,kernel_size=2,stride=2)
        self.dense=nn.Linear(in_features=90,out_features=1)
        self.batch_transform=batch_transform
        self.loss=loss

    def forward(self,x):

        x=self.batch_transform(x) # eg. rescale
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=torch.flatten(x,start_dim=1)
        x=torch.sigmoid(self.dense(x))

        return x
