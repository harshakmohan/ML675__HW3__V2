""" Model classes defined here! """

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(784,hidden_dim) #784 input features, hidden_dim output features
        self.layer2 = nn.Linear(hidden_dim, 10) #hidden_dim input features, 10 output features
        self.model = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.Softmax(dim=1))

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        return self.model(x)

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_chan, kernel_size=n1_kern, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_chan, out_channels=10, kernel_size=n2_kern, stride=2)
        self.pooling = torch.nn.MaxPool2d(kernel_size=((29 - n1_kern) - n2_kern)//2 + 1)
        self.simple_cnn = None

    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], 1, 28, 28))
        self.simple_cnn = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True), self.pooling, nn.Softmax(dim=1))
        return self.simple_cnn(x)

class BestNN(torch.nn.Module):    # take hyperparameters from the command line args!
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super(BestNN, self).__init__()

    def forward(self, x):
        return x