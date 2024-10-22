import torch
import torch.nn as nn





class ConvLayer(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super().__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(in_channels=inputChannels, out_channels=outputChannels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=outputChannels, out_channels=outputChannels, kernel_size=3, padding=1, stride=1),#Run 2 Convolutional layers through the model
            nn.ReLU(),
        )

    def forward(self, X):
        return self.ConvLayer(X)
    



class MaskModel(nn.Module):
    def __init__(self, inputChannels, outputChannels, sizes):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()#Used to create the paths between both sides of the 'U'
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        for size in sizes:
            self.downs.append(ConvLayer(inputChannels, size))#adding each conv block to the left side of the U
            inputChannels = size

        for size in reversed(sizes):#Have to go from small to big
            self.ups.append(nn.ConvTranspose2d(in_channels=size*2,#This is because its counter part on the the side is twice as large
                                                out_channels=size, #Shrinking cause we are taking less features,
                                                kernel_size=2, stride=2))
            self.ups.append(ConvLayer(inputChannels=size*2, outputChannels=size))#Run it through the layer again to make sure that the finer details are captured
        
        self.bridge = ConvLayer(sizes[-1], sizes[-1]*2)#Bottem part of the U
        self.finalLayer = nn.Conv2d(sizes[0], outputChannels, kernel_size=1)#Creating the mask



    def forward(self, X):
        paths = []#holds the connections between the ups and downs
        for down in self.downs:
            X=down(X)
            paths.append(X)
            X = self.maxPool(X)
        
        X = self.bridge(X)#reached the bottom of the 'U'

        paths = list(reversed(paths))#reverse the list to me life easy

        for i in range(0, len(self.ups), 2):# need to take 2 steps as you want to hit both the transpose and the conv layer
            X = self.ups[i](X)
            path = paths[i//2]#need to make sure that it is an int
            catSkip = torch.cat((path, X), dim=1)#has to be dim=1 as that is the dimension of the channels as it is batch, channel, height, width
            X= self.ups[i+1](catSkip)#Goes through the ConvLayer if the new transposed matrix
        
        return self.finalLayer(X)
            

