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

    layerToIndex = {
        1: "RB",
        2: "CB",
        3: "MP",
        4: "AP",
        5: "BND",
        6: "CON",
        7: "SUM",
        8: "OUT"
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
    
    def decode (encodedArch):
        decodedArch = []
        for block in range (0, len(encodedArch), 5):
            layerType = ArchitectureCodec.layerToIndex[encodedArch[block]]
            # Try loop is here for when it gets to the output layer which does not have kernal and filter size parameters
            # This should only occur when it is the output layer
            try:
                filerSize = ArchitectureCodec.filterSize[encodedArch[block + 3]]
                kernalSize = ArchitectureCodec.kernalSize[encodedArch[block + 4]]

                decodedLayer = {
                    "type": layerType,
                    "Connection 1": encodedArch[block + 1],
                    "Connection 2": encodedArch[block + 2],
                    "Filter Size": filerSize,
                    "Kernal Size": kernalSize
                }
            except IndexError:
                # Just a sanity check incease it does get to this error and it is not an output layer
                if layerType != 'OUT':
                    print("Error: Invalid layer type in encoded architecture")
                    continue

                decodedLayer = {
                    "type": layerType,
                    "Connection 1": encodedArch[block + 1]
                }
            decodedArch.append(decodedLayer) # Add decoded layer to the array

        return decodedArch
    
    def activeLayers (architecture):
        activeLayers = np.zeros(len(architecture), dtype=bool) # Create array with all false values
        # Get the outputlayer connection layer and check it as a active layer
        activeLayers[len(architecture) - 1] = True
        
        outputConnection = architecture[len(architecture) - 1]["Connection 1"]
        connectedLayer = architecture[outputConnection]

        ArchitectureCodec.checkConnection(architecture, outputConnection, connectedLayer, activeLayers)
        for i in range (len(activeLayers)):
            print(f"Layer {i}: {architecture[i]}, Active: {activeLayers[i]}")

    def checkConnection (architecture, layerIndex, connectedLayer, activeLayers):
        # Set the connected layer as active
        activeLayers[layerIndex] = True

        # Check the first connected layer of each layer. If it is sum / concat layer then also check the second connected layer
        if layerIndex != 0:
            connectedNode = connectedLayer["Connection 1"] # Get the connection node from the layer parameters
            ArchitectureCodec.checkConnection(architecture, connectedNode, architecture[connectedNode], activeLayers)

            if connectedLayer["type"] == 'CON' or connectedLayer["type"] == 'SUM':
                connectedNode2 = connectedLayer["Connection 2"]
                ArchitectureCodec.checkConnection(architecture, connectedNode2, architecture[connectedNode2], activeLayers)