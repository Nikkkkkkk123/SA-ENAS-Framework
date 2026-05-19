import torch.nn as nn

# Taken from https://github.com/sg-nm/cgp-cnn-PyTorch/blob/master/cnn_model.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class LinearBlock (nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Sequential(nn.Flatten(),
            nn.Linear((in_features * 28 * 28), out_features))
    def forward(self, x):
        x = self.linear(x)
        return x

class model (nn.Module):
    def __init__(self, architecture, noClasses):
        super(model, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input = 1 # Grayscale input 
        for layer in architecture:
            if layer["type"] == "CB":
                self.layers.append(ConvBlock(self.input, layer["Filter Size"], layer["Kernel Size"]))
                self.input = layer["Filter Size"]
            elif layer["type"] == "RB":
                # Implement ResNet block here
                pass
            elif layer["type"] == "OUT":
                # Implement output layer here
                pass
            elif layer["type"] == "IN":
                pass
        self.layers.append(LinearBlock(self.input, noClasses))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x