import torch.nn as nn
import torch

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
        x = self.conv(x)
        return x

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

class Sum (nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, input1, input2):
        
        if input1.shape[1] != input2.shape[1]:
            if input1.shape[1] < input2.shape[1]:
                input1 = nn.functional.pad(input1, (0, 0, 0, 0, 0, input2.shape[1] - input1.shape[1]))
            else:
                input2 = nn.functional.pad(input2, (0, 0, 0, 0, 0, input1.shape[1] - input2.shape[1]))
        output = torch.add(input1, input2)
        return output

class Con (nn.Module):
    def __init__(self):
        super(Con, self).__init__()

    def forward(self, input1, input2):
        # Currently a place hodler and untested
        output = torch.cat((input1, input2), dim=1)
        return output


class LinearBlock (nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Sequential(nn.Flatten(),
            nn.Linear(in_features * 28 * 28, out_features)) # This is currently hard coded to the test image size
    def forward(self, x):
        x = self.linear(x)
        return x

class model (nn.Module):
    """
    Function Name: __init__
    Description: This is the constructure function for the model class. It takes an input architecture and initialises the corresponding architecture
    Parameter: 
        architecture: The architecture to be initialised
        noClasses: The number of classes that the dataset uses
        inputChannels: The number of color channels in the input images. This is needed to initialise the first layer correctly.
    Return: 
        None
    """
    def __init__(self, architecture, noClasses, inputChannels):
        super(model, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        # Due to architecture generation allowing for layers to go to different layers at once, filter sizes need to be stored so they can be obtained as the 8th layer may still take input from the 1st layer
        self.layerSizes = [inputChannels] 
        # Loop through each layer (skipping the input layer as it will cause an error)
        for layer in architecture[1:]:
            archLayer = self.layerSwitch(layer, noClasses)

            # I like the switch case being used but if it is an input layer it returns none causing an error
            # while the for loop should skip this with [1:], this is just a sanity check to ensure that if it return none it will skip adding it
            if archLayer is not None:
                self.layers.append(archLayer)
    
    """
    Function Name: layerSwitch
    Description: This is a helper function that takes the layer being intialised and returns the corresponding layer.
    Parameter: 
        layer: The layer in the architecture being initalisaed
        noClasses: The number of classes that the dataset uses
    Return: 
        generatedLayer: This is the initialised layer the architecture
    """
    def layerSwitch (self, layer, noClasses):
        connectionSize = self.getConnectionSize(layer["Connection 1"])
        match layer["type"]:
            case "CB":
                generatedlayer = ConvBlock(connectionSize, layer["Filter Size"], layer["Kernel Size"])
                self.layerSizes.append(layer["Filter Size"]) # This is to update the input for the next layer
                return generatedlayer
            case "RB":
                generatedlayer = ResBlock(connectionSize, layer["Filter Size"], layer["Kernel Size"])

                """
                ResBlocks have skip connection. Bassically if doing the same as the convblock when
                self.input > filter size it would store the smaller output size and therefore cause an 
                error as the next layer would be expecting the smaller size
                """
                self.layerSizes.append(max(connectionSize, layer["Filter Size"]))
                return generatedlayer
            case "SUM":
                # Currently not tested
                # Get the two connected layers so that they can be summed together
                Layer1 = self.architecture[layer["Connection 1"]]
                Layer2 = self.architecture[layer["Connection 2"]]

                generatedLayer = Sum().forward(Layer1, Layer2)

                self.layerSizes.append(max(self.getConnectionSize(layer["Connection 1"]), self.getConnectionSize(layer["Connection 2"]))) # Same as "RB". get the max so that the right sized output is being used
                return generatedLayer
            case "CON":
                # Filler. currently not impelemented
                return None 
            case "OUT":
                return LinearBlock(connectionSize, noClasses)
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
    def getConnectionSize (self, layer):
        return self.layerSizes[layer]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x