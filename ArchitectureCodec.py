import numpy as np

class ArchitectureCodec:
    layerType = {
        "RB": 1,
        "CB": 2,
        "MP": 3,
        "AP": 4,
        "BND": 5,
        "CON": 6,
        "SUM": 7,
        "OUT": 8
    }

    kernalSize = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    filterSize = [0, 8, 16, 32, 64, 128, 256, 512]

    def encode (architecture):
        encodedArch = []
        
        # Encode each layer of the architecture
        for i in range (len(architecture)):
            layer = architecture[i]
            encodedType = ArchitectureCodec.layerType[layer["type"]]

            """
                Have to try to encode the filter size and kernal size as the output layer does not include these in the parameters
                This is included with the encoding formation of layer = [Function type | Connection 1 | Connection 2 | Filter Size | Kernel Size]
                For the architecture encoding it follows [Layer 1 | Layer 2 | ... | Layer n] where n is the output layer following the encoding of 
                Output = [Function type | connection 1]
            """
            try:
                encodedFilterSize = ArchitectureCodec.filterSize.index(layer["Filter Size"])
                encodedKernalSize = ArchitectureCodec.kernalSize.index(layer["Kernal Size"])
                encodedLayer = [encodedType, layer["Connection 1"], layer["Connection 2"], encodedFilterSize, encodedKernalSize]
            except KeyError:
                encodedLayer = [encodedType, layer["Connection 1"]]
            
            encodedArch.append(encodedLayer) # Encode the layer parameters and add them to the encoded architecture
        
        # Flatten the encoded architecture to a 1D array for input into the surrogate model
        encodedArch = [layer for sublist in encodedArch for layer in sublist]
        
        return encodedArch
            