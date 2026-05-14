# Going to use the paper with the ConvBlock and ResNet block to get the mix of layers and parameters that I want
# ConvBlock (CB): Conv2D -> BatchNorm -> ReLU
# Encoding for this would be [CB | Connection 1 | Connection 2 | Filter Size | Kernal Size] (This is due to it using a stride of 1
# ResBlock (RB): ConvBlock -> Convolution -> BachNorm -> Sum -> ReLu [RB | Connection 1 | Connection 2 | Filter size | kernal Size]
import random
import numpy as np
import matplotlib.pyplot as plt

from GenerateLayers import GenerateLayers as gl

class GenerateArchitecture:
   
    noLayerTypes = 7
    noParam = 5

    # Function Name: generateArchitecture
    def generateArchitecture (length):
        architecture = []
        
        inputLayer = {
            "type": "IN",
        }
        architecture.append(inputLayer)
        # Loop through the number of layers to generate the required amount to have a full size architecture
        for i in range (length - 1):
            # Select a random block type to generate
            # The int is converted to a block type in the generateLayer function
            blockType = random.randint(1, GenerateArchitecture.noLayerTypes)

            # Generate a layer and store it in the architecture array
            architecture.append(gl.generateLayer(blockType, i))
        # Generate the output node and connect it to a random node in the architecture that isnt the inputnode or itself
        architecture.append(gl.generateOutputLayer(layerIndex = (length - 2))) # This has to be -2 as it is included currently in the architecture length. So -1 to ensure it cant pick itself and -1 due to having length -1 architecture size at this point

        return architecture