import torch.nn as nn
import torch
import numpy as np
import os
import math
from Node import Node

# Taken from https://github.com/sg-nm/cgp-cnn-PyTorch/blob/master/cnn_model.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        output = self.conv(x)
        return output

class ResBlock (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2

        # The diagram in "Evolution of Deep Convolutional Neural Networks Using Cartesian Genetic Programming" (https://doi.org/10.1162/evco_a_00253)
        """
        The diagram is:
                Input -> ConvBlock -> Convolution -> BatchNorm -> Sum -> Relu
                  |                                                ^  
                   ------------------------------------------------|
        """
        self.convBlock = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        # If input and output channels are different then the shapes cannot be summed together.
        # This can be solved either with a conv1x1 or padding. We are initially going to just do padding 
        # but testing should be completed to see if 1x1 is viable or if potentially have two resblock options with padding and one with conv1x1
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        SpareInput = input
        x = self.convBlock(input)
        x = self.conv(x)

        output = Sum().forward(SpareInput, x)
        x = self.relu(output)

        return x

class MaxPool (nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(x)
        return x

class AvgPool (nn.Module):
    def __init__(self, kernel_size, stride):
        super(AvgPool, self).__init__()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.pool(x)
        return x

class Sum (nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, input1, input2):

        '''
        If there are differnet dimension sizes then they need to be modified so that they are the same.
        If the number of channels if different you can either:
            - Pad the smaller one with zeros
            - 1x1 conv to change the number of channels to match

        if the height / width are different then you can either:
            - pad the smaller dimension with zeros
            - or use pooling to reduce the larger dimension to match the smaller
        '''
        if input1.shape[1] != input2.shape[1]:
            if input1.shape[1] < input2.shape[1]:
                input1 = nn.Conv2d(input1.shape[1], input2.shape[1], kernel_size=1)(input1)
            else:
                input2 = nn.Conv2d(input2.shape[1], input1.shape[1], kernel_size=1)(input2)

        # To determine the number of pooling layers required if the dimensions are different,
        # the formula for number of devisions is utilised
        # No Pooling = log2 (larger dimension / smaller dimension)
        if input1.shape[2] != input2.shape[2]:
            if input1.shape[2] < input2.shape[2]:
                noPools = math.floor(math.log2(input2.shape[2] / input1.shape[2]))
                for i in range(noPools):
                    input2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(input2)
            else:
                noPools = math.floor(math.log2(input1.shape[2] / input2.shape[2]))
                for i in range(noPools):
                    input1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(input1)
        output = torch.add(input1, input2)
        return output

class Con (nn.Module):
    def __init__(self):
        super(Con, self).__init__()

    def forward(self, input1, input2):
        # Currently an error if the tensor sizes do not match on other dimensions other than 1.
        # This can be solved either by padding or pooling.
        # Since architecture size is a particial objective pooling should be used to reduce computaitonal cost. 
        # The idea for pooling for me came from "https://github.com/sg-nm/cgp-cnn-PyTorch/blob/master/cnn_model.py#L278" -> this idea adds multiple pooling layers until it gets to the size
        # However adding this pooling layer would change the architecture encoding. So it should be encoded differently now.
        # currently not going to change the encoding structure but this should be mentioned

        # If dimenion 2 of the image is different between the 2 inputs then the larger one should be pooled
        if input1.size(2) > input2.size(2):
            # you can determine the required filter size by going FilterSize = bigInput - ((small - 1) * stride)
            # for small = 14 big = 64, stride = 2 it would be:
            # filtersize = 64 - ((14 - 1) * 2) = 64 - ((13) * 2) = 64 - 26
            # filtersize = 38
            # For now just going to do multiple pooling layers
            # Just doing normal 2 filter and 2 stride just halves the size so can just do this following the git
            # But if it was the same example as before this could cause an error as 64 does not evenly go into 14 and it would equal in 16 sized dimension
            numberPools = math.floor(math.log2(input1.size(2) / input2.size(2)))
            for i in range(numberPools):
                input1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(input1)
        elif input1.size(2) < input2.size(2):
            numberPools = math.floor(math.log2(input2.size(2) / input1.size(2)))
            for i in range(numberPools):
                input2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(input2)

        output = torch.cat((input1,  input2), dim=1)
        return output


class LinearBlock (nn.Module):
    def __init__(self, in_features, out_features, imageDimension):
        super(LinearBlock, self).__init__()
        self.linear = nn.Sequential(
            # for some reasin this is adding up wrong?
            # Image width and height is being calculated incorrectly
            nn.Linear(in_features * imageDimension * imageDimension, out_features)
        ) 
    def forward(self, x):
        x = self.linear(x)
        return x

class model (nn.Module):
    """
    Function Name: __init__
    Description: This is the constructure function for the model class. It takes an input architecture and initialises the corresponding architecture
    Parameter: 
        architecture: The architecture object to be initialised
        noClasses: The number of classes that the dataset uses
        inputChannels: The number of color channels in the input images. This is needed to initialise the first layer correctly.
    Return: 
        None
    """
    def __init__(self, architecture, noClasses, inputChannels):
        super(model, self).__init__()
        self.layers = nn.ModuleList()

        self.architecture = architecture
        # print("Full Architecture", end=" ")
        # architecture.print()
        # print(f"Active architectre {architecture.getActiveArch()}")

        self.numberClasses = noClasses

        # Due to architecture generation allowing for layers to go to different layers at once, filter sizes need to be stored so they can be obtained as the 8th layer may still take input from the 1st layer
        self.layerSizes = {}
        self.layerSizes[0] = 1
        index = 1
        self.imageDimensions = {}
        self.imageDimensions[0] = 28 # currently the width and height of the input image is hard coded and needs to be changed
        self.activeArch = list(architecture.getActiveArch())
        for layer in list(architecture.getActiveArch().values())[1:]:
            archLayer = self.layerSwitch(layer, index)

            # I like the switch case being used but if it is an input layer it returns none causing an error
            # while the for loop should skip this with [1:], this is just a sanity check to ensure that if it return none it will skip adding it
            if archLayer is not None:
                self.layers.append(archLayer)
                index += 1
            
    """
    Function Name: layerSwitch
    Description: This is a helper function that takes the layer being intialilayer["Kernel Size"]sed and returns the corresponding layer.
    Parameter: 
        layer: The layer in the architecture being initalisaed
    Return: 
        generatedLayer: This is the initialised layer the architecture
    """
    def layerSwitch (self, layer, index):
        connectionSize = layer.getConnectionOutputSize(1)
        match layer.getNodeType():
            case "CB":
                generatedlayer = ConvBlock(layer.getConnectionOutputSize(1), layer.getFilterSize(), layer.getKernelSize())
                return generatedlayer
            case "RB":
                generatedlayer = ResBlock(layer.getConnectionOutputSize(1), layer.getFilterSize(), layer.getKernelSize())

                """
                ResBlocks have skip connection. Bassically if doing the same as the convblock when
                self.input > filter size it would store the smaller output size and therefore cause an 
                error as the next layer would be expecting the smaller size
                """
                return generatedlayer
            case "SUM":
                generatedLayer = Sum()                
                return generatedLayer
            case "CON":
                generatedLayer = Con()
                return generatedLayer
            case "MP" | "AP":
                generatedLayer = MaxPool() if layer.getNodeType() == "MP" else AvgPool(layer.getKernelSize(), layer.getKernelSize())
                return generatedLayer
            case "LIN":
                return LinearBlock(layer.getConnectionOutputSize(1), self.numberClasses, layer.getImageDimension())
            case "IN":
                return None # This is just a skip as there is no layer for the input layer

    """
    Function Name: getConnectionSize
    Description: This is a helper function that returns the output size of the specified connected layer.
    Parameter: 
        layer: The layer index for the connected layer.
    Return: 
        self.layerSizes[layer]: The output size of the connected layer.
    """
    def getConnectionSize (self, layer: Node, connectionNodeID):
        # This needs to get the connection size of the connected layer
        connectionNode = layer.getConnection1() if connectionNodeID == 1 else layer.getConnection2()
        return connectionNode.getLayerSize()
    
    def _getOutputConnection (self, output, layerIndex, connectNodeID):
        # This will return the output shape from the required input layer
        # To do this i need to:
        #   Use the connectedNodeID to know whether it is connection 1 or 2
        #   use this to get the index from the active layers from the selected connection node
        #   then this needs to return the output from this index
        if connectNodeID == 1:
            connectionNode = self.architecture.getNode(layerIndex).getConnection1().getNodeId()
        else:
            connectionNode = self.architecture.getNode(layerIndex).getConnection2().getNodeId()

        return output[connectionNode]
        
    
    def main (self, x):
        output = {}
        output[0] = x
        index = 1

        for layer in self.layers:
            layerIndex = self.activeArch[index]
            connection1Output = self._getOutputConnection(output, layerIndex, 1)
            if isinstance(layer, ConvBlock) | isinstance(layer, ResBlock):
                # Need to get the connected layers index in the output array so that if one layer outputs to multiple then it is using the correct input size
                output[layerIndex] = layer(connection1Output)
            elif isinstance(layer, LinearBlock):
                temp = connection1Output.view(connection1Output.size(0), -1)
                output[layerIndex] = layer(temp)
            elif isinstance(layer, Con) | isinstance(layer, Sum):
                layerIndex = self.activeArch[index]
                connection2Output = self._getOutputConnection(output, layerIndex, 2)
                output[layerIndex] = layer(connection1Output, connection2Output)
            elif isinstance(layer, MaxPool) | isinstance(layer, AvgPool):
                if connection1Output.size(2) >= 1:
                    output[layerIndex] = layer(connection1Output)
                else:
                    output[layerIndex] = connection1Output            
            index += 1
        return output[self.activeArch[-1]]

    def forward(self, x):
        return self.main(x)