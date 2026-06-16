import numpy as np
import os
import copy

class ArchitectureCodec:
    layerType = {
        "RB": 1,
        "CB": 2,
        "MP": 3,
        "AP": 4,
        "BND": 5,
        "CON": 6,
        "SUM": 7,
        "LIN": 8
    }

    layerToIndex = {
        1: "RB",
        2: "CB",
        3: "MP",
        4: "AP",
        5: "BND",
        6: "CON",
        7: "SUM",
        8: "LIN"
    }

    kernelSize = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    filterSize = [0, 8, 16, 32, 64, 128, 256, 512]

    def encode (architecture):
        encodedArch = []
        
        # Encode each layer of the architecture
        for i in range (1, len(architecture)):
            layer = architecture[i]
            encodedType = ArchitectureCodec.layerType[layer["type"]]

            """
                Have to try to encode the filter size and kernel size as the output layer does not include these in the parameters
                This is included with the encoding formation of layer = [Function type | Connection 1 | Connection 2 | Filter Size | Kernel Size]
                For the architecture encoding it follows [Layer 1 | Layer 2 | ... | Layer n] where n is the output layer following the encoding of 
                Output = [Function type | connection 1]
            """
            try:
                encodedFilterSize = ArchitectureCodec.filterSize.index(layer["Filter Size"])
                encodedKernelSize = ArchitectureCodec.kernelSize.index(layer["Kernel Size"])
                encodedLayer = [encodedType, layer["Connection 1"], layer["Connection 2"], encodedFilterSize, encodedKernelSize]
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
            # Try loop is here for when it gets to the output layer which does not have kernel and filter size parameters
            # This should only occur when it is the output layer
            try:
                filerSize = ArchitectureCodec.filterSize[encodedArch[block + 3]]
                kernelSize = ArchitectureCodec.kernelSize[encodedArch[block + 4]]

                decodedLayer = {
                    "type": layerType,
                    "Connection 1": encodedArch[block + 1],
                    "Connection 2": encodedArch[block + 2],
                    "Filter Size": filerSize,
                    "Kernel Size": kernelSize
                }
            except IndexError:
                # Just a sanity check incease it does get to this error and it is not an output layer
                if layerType != 'LIN':
                    print("Error: Invalid layer type in encoded architecture")
                    continue

                decodedLayer = {
                    "type": layerType,
                    "Connection 1": encodedArch[block + 1]
                }
            decodedArch.append(decodedLayer) # Add decoded layer to the array

        return decodedArch
    
    def activeLayers (architecture):
        isactiveLayer = np.zeros(len(architecture), dtype=bool) # Create array with all false values
        # Get the outputlayer connection layer and check it as a active layer
        isactiveLayer[len(architecture) - 1] = True
        activeArchitecture = []
        alteredLayers = {}

        outputConnection = architecture[len(architecture) - 1]["Connection 1"]
        connectedLayer = architecture[outputConnection]

        ArchitectureCodec.checkConnection(architecture, outputConnection, connectedLayer, isactiveLayer)

        activeArchitecture.append(architecture[0])
        ArchitectureCodec._orignalToAlteredLayer(alteredLayers, 0, 0)
        for layerIndex in range (1, len(architecture) - 1):
            if isactiveLayer[layerIndex]:
                repairdLayer = ArchitectureCodec._activeLayerRepair(architecture, alteredLayers, activeArchitecture, layerIndex)
                activeArchitecture.append(repairdLayer)

                # Use the index of the repaired active layer to store the index of the original unaltered way so that it can be accessed
                ArchitectureCodec._orignalToAlteredLayer(alteredLayers, layerIndex, (len(activeArchitecture) - 1))

        activeArchitecture.append(ArchitectureCodec._repaireFinalActiveLayer(architecture, alteredLayers, (len(architecture) - 1)))
        return activeArchitecture
    
    def _activeLayerRepair (architecture, alteredLayers, activeArch, layerIndex):
        currLayer = architecture[layerIndex]
        
        # Since the layer connection information changes it wont be able to find it in the activeArch array. So have to get the index from a dictionary
        # that stores the original layer index with the repaired
        newCon1Index = ArchitectureCodec._getAlteredLayerIndex(alteredLayers, currLayer["Connection 1"])
        
        # If this is a laytype which only cares about the first connectin then the second may not be active. In this case just store the original
        try:
            newCon2Index = ArchitectureCodec._getAlteredLayerIndex(alteredLayers, currLayer["Connection 2"])
        except:
            newCon2Index =architecture[currLayer["Connection 2"]]

        repairedLayer = copy.copy(currLayer)
        repairedLayer["Connection 1"] = newCon1Index
        repairedLayer["Connection 2"] = newCon2Index

        return repairedLayer
    
    def _repaireFinalActiveLayer (architecture, alteredLayers, layerIndex):
        # Since fully connected layers only take 1 input this would have to mean that the last index of the active architectures is its input. Since 
        # if there is anything defined after it then its either the input or a combination block that would become an input
        layer = architecture[layerIndex]
        repairdLayer = copy.copy(layer)

        connection = layer["Connection 1"]

        activeConnection = ArchitectureCodec._getAlteredLayerIndex(alteredLayers, connection)

        repairdLayer["Connection 1"] = activeConnection
        return repairdLayer
    
    def _orignalToAlteredLayer (alteredLayers, repairedLayerIndex, originalIndex):
        alteredLayers[repairedLayerIndex] = originalIndex
    
    def _getAlteredLayerIndex (alteredLayers, originalIndex):
        return alteredLayers[originalIndex]

    def checkConnection (architecture, layerIndex, connectedLayer, isactiveLayer):
        # Set the connected layer as active
        isactiveLayer[layerIndex] = True

        # Check the first connected layer of each layer. If it is sum / concat layer then also check the second connected layer
        if layerIndex != 0:
            connectedNode = connectedLayer["Connection 1"] # Get the connection node from the layer parameters
            ArchitectureCodec.checkConnection(architecture, connectedNode, architecture[connectedNode], isactiveLayer)

            if connectedLayer["type"] == 'CON' or connectedLayer["type"] == 'SUM':
                connectedNode2 = connectedLayer["Connection 2"]
                ArchitectureCodec.checkConnection(architecture, connectedNode2, architecture[connectedNode2], isactiveLayer)
    
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