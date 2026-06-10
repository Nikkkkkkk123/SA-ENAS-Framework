# Going to use the paper with the ConvBlock and ResNet block to get the mix of layers and parameters that I want
# ConvBlock (CB): Conv2D -> BatchNorm -> ReLU
# Encoding for this would be [CB | Connection 1 | Connection 2 | Filter Size | Kernal Size] (This is due to it using a stride of 1
# ResBlock (RB): ConvBlock -> Convolution -> BachNorm -> Sum -> ReLu [RB | Connection 1 | Connection 2 | Filter size | kernal Size]
import random
import numpy as np
import matplotlib.pyplot as plt

from GenerateLayers import GenerateLayers as gl
from Architecture import Architecture as arch
import os

class GenerateArchitecture:
   
    noLayerTypes = 7
    noParam = 5

    """ 
    Function Name: generateArchitectures
    Description: This function loops to generate the requried number of architectures and stores them in an array to be returned
    Parameter: 
        noArchs: The number of architectures to generate
        length: The length of the architecture to generate (including the output layer)
    Return: 
        architectures: An array of architectures
    """
    def generateArchitectures (noArchs, length):
        architectures = []

        for i in range (noArchs):
            generatedArchitecture = GenerateArchitecture.generateArchitecture(length)
            if GenerateArchitecture.checkDuplicates(generatedArchitecture, architectures):
                print("Duplicate architecture found. Generating a new architecture.")
                while GenerateArchitecture.checkDuplicates(generatedArchitecture, architectures):
                    generatedArchitecture = GenerateArchitecture.generateArchitecture(length)
            architectures.append(generatedArchitecture)
        os._exit(0)
        return architectures
    
    """
    Function Name: checkDuplicates
    Description: This is a function to check if the generated architecture is a duplicate of any that already exist in the population.
    Parameter:
        newArch: The architecture to check for duplicates
        architectures: The list of existing architectures to check against
    Return:
        boolean: True if a duplicate was found, otherwise it will return false.
    """
    def checkDuplicates (newArch, architectures):
        for arch in architectures:
            if newArch.getActiveArch() == arch.getActiveArch():
                return True
        return False
    """
    Function Name: generateArchitecture
    Description: This function generates a single architecture of the required length and then returns it
    Parameter:
        length: The length of the architecture to generate (including the output layer)
    Return:
        architecture: an array of layers for a single full architecture
    """
    def generateArchitecture (length):
        generatedarchitecture = []

        # Currently implemented as a safe guard for finding active layers
        inputLayer = {
            "type": "IN",
        }
        generatedarchitecture.append(inputLayer)
        # Loop through the number of layers to generate the required amount to have a full size architecture
        for i in range (length - 1):
            # Select a random block type to generate
            # The int is converted to a block type in the generateLayer function
            blockType = random.randint(1, GenerateArchitecture.noLayerTypes)

            # Generate a layer and store it in the architecture array
            generatedarchitecture.append(gl.generateLayer(blockType, i))
        # Generate the output node and connect it to a random node in the architecture that isnt the inputnode or itself
        generatedarchitecture.append(gl.generateOutputLayer(layerIndex = (length - 1))) # This has to be -2 as it is included currently in the architecture length. So -1 to ensure it cant pick itself and -1 due to having length -1 architecture size at this point

        # Creates an architecture object with the generated architecture structure
        architecture = arch(generatedarchitecture) 

        return architecture