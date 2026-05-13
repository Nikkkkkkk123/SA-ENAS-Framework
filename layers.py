# Going to use the paper with the ConvBlock and ResNet block to get the mix of layers and parameters that I want
# ConvBlock (CB): Conv2D -> BatchNorm -> ReLU
# Encoding for this would be [CB | Connection 1 | Connection 2 | Filter Size | Kernal Size] (This is due to it using a stride of 1
# ResBlock (RB): ConvBlock -> Convolution -> BachNorm -> Sum -> ReLu [RB | Connection 1 | Connection 2 | Filter size | kernal Size]
import random
import numpy as np
import matplotlib.pyplot as plt

class Layer:
    # Input stem options

    # Below are three block types that can be selected from
    ResBlock = {
        "type": "RB",
        "Connection 1": None,
        "Connection 2": None,
        "Filter Size": [8, 16, 32, 64, 128, 256, 512],
        "Kernal Size": [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }

    ConvBlock = {
        "type": "CB",
        "Connection 1": None,
        "Connection 2": None,
        "Filter Size": [8, 16, 32, 64, 128, 256, 512],
        "Kernal Size": [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }

    maxPool = {
        "type": "MP",
        "Connection 1": None,
        "Connection 2": None,
        "Filter Size": None,
        "Kernal Size": [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }

    averagePool = {
        "type": "AP",
        "Connection 1": None,
        "Connection 2": None,
        "Filter Size": None,
        "Kernal Size": [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }

    bottleNeckDepthWise = {
        "type": "BND",
        "Connection 1": None,
        "Connection 2": None,
        "Filter Size": [8, 16, 32, 64, 128, 256, 512],
        "Kernal Size": [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }

    concat = {
        "type": "CON",
        "Connection 1": None,
        "Connection 2": None,
    }

    summation = {
        "type": "SUM",
        "Connection 1": None,
        "Connection 2": None,
    }

    output = {
        "type": "OUT",
        "Connection 1": None
    }

    layerBlocks = {
        1: ResBlock,
        2: ConvBlock,
        3: maxPool,
        4: bottleNeckDepthWise,
        5: averagePool,
        6: concat,
        7: summation,
        8: output
    }

    noParam = 5

    def __init__ (self, length):
        self.length = length
        self.active = np.zeros(length, dtype=bool)
        self.decodedArch = np.zeros((length, self.noParam)) 
        pass

    def generate_architecture (self):
        length = self.length

        numpyArch = np.zeros((length, self.noParam)) # Create a numpy array to store the architecture in a format that is easier to manipulate. Each row represents a block and each column represents a parameter of the block (type, connection 1, connection 2, filter size, kernal size)
        
        # Loop to create each block of the architecture
        # This includes selecting a block type and required connection node and parameters
        for i in range (length - 1):
            # Select a random block type from the available
            # This is done by selecting between the first index to the second last of layerblocks
            blockType = random.randint(1, len(self.layerBlocks) - 1)

            # Generate the layer and store it in the array
            layer = self.generateBlock(blockType, 0 if i == 0 else i - 1) # Generate the block with the selected block type and the current layer index to ensure valid connections
            numpyArch[i] = [blockType, layer["Connection 1"], layer["Connection 2"], layer["Filter Size"], layer["Kernal Size"]] # Add the block parameters to the numpy array
        
        # Generate the output node and connect it to a random node in the architecture that isnt the inputnode or itself
        outputNode = self.generateBlock(8, length - 1) # Generate the output node
        
        # Manually create the output layer
        numpyArch[length - 1] = [8, outputNode["Connection 1"], None, None, None]

        self.decodedArch = self.decodeArch(numpyArch) # Decode the architecture to a human readable format

        self.activeLayers(self.decodedArch) # Get the active layers in the architecture
        return self.decodedArch, self.active

    def generateBlock (self, blockID, layerIndex):

        match blockID:
            case 1 | 2 | 3:
                block = self.layerBlocks[blockID] # Get the block from the dictionary

                # Select the connection nodes. Each node can only have 2 connections maximum.
                connectionNode1 = random.randint(0, layerIndex) # Randomly select a connection node for the first connection
                connectionNode2 = random.randint(0, layerIndex) # Randomly select a connection node for the second connection

                filterSize = random.randint(1, 8) # Randomly selects the index for the filter size. This is to make it easier for the RBF surrogate as it deals with euclidean distance
                kernalSize = random.randint(1, 9) 
            case 4 | 5:
                block = self.layerBlocks[blockID] # Get the block from the dictionary

                connectionNode1 = random.randint(0, layerIndex) # Randomly select a connection node for the first connection
                connectionNode2 = random.randint(0, layerIndex) # Randomly select a connection node for the second connection

                filterSize = 0
                kernalSize = random.randint(1, 9)
            case 6 | 7:
                # To stop Sum / Concat being the first block it will be forced to require at least 2 nodes to connect too
                if layerIndex < 2:
                    blockID = random.randint(1, 5) # Randomly select a block type that isnt Sum or Concat
                    return self.generateBlock(blockID, layerIndex) # Regenerate the block with the new block ID

                block = self.layerBlocks[blockID] # Get the block from the dictionary
                
                connectionNode1 = random.randint(0, layerIndex) # Randomly select a connection
                connectionNode2 = random.randint(0, layerIndex) # Randomly select a connection
                while connectionNode2 == connectionNode1: # Ensure the second connection node is different to the first
                    connectionNode2 = random.randint(0, layerIndex) # Randomly select a connection node for the second connection
                
                filterSize = 0
                kernalSize = 0

            case 8:
                # Randomly select a connection node for the output layer. It can be anywhere from the\
                connectionNode1 = random.randint(0, (layerIndex - 1))
                
                layer = {
                    "type": blockID,
                    "Connection 1": connectionNode1
                }
                
                return layer

        # Store the layer to be returned
        layer = {
            "type": blockID,
            "Connection 1": connectionNode1,
            "Connection 2": connectionNode2,
            "Filter Size": filterSize,
            "Kernal Size": kernalSize
        }
        return layer
    
    def decodeArch (self, architecture):
        decodedArch = [] # Create an empty numpy array to store the decoded architecture. Each row represents a block and each column represents a parameter of the block (type, connection 1, connection 2, filter size, kernal size)
        for i in range (len(architecture)):
            block = architecture[i]
            layerNo = int(i) # Get the index of the block in the architecture to use as the layer number
            layer = self.layerBlocks[block[0]]["type"]
            connection1 = int(block[1])
            
            if not np.isnan(block[2]):
                connection2 = int(block[2])
                filterIndex = int(block[3])
                kernalIndex = int(block[4])

                decodedArch.append([layerNo, layer, connection1, connection2, filterIndex, kernalIndex]) # Decode the block parameters and add them to the decoded architecture
            else:
                decodedArch.append([layerNo, layer, connection1, None, None, None])
        decodedArch = np.array(decodedArch) # Convert the decoded architecture to a numpy array for easier manipulation
        return decodedArch

    def activeLayers (self, architecture):
        # To check if a layer is active the connection 1 node of the output layer which should always be the last in the array
        # is to be checked
        outputLayer = architecture[len(architecture) - 1] # Get the output layer which is the last layer in the architecture
        self.active[outputLayer[0]] = True # Set the output connection node as active
        print(architecture)
        self.checkConnection(architecture[outputLayer[2]]) # Since it starts from the output layer only check the single one
        
        print("Active Layers:")
        for i in range (len(self.active)):
            if self.active[i] == True:
                print(self.decodedArch[i])

    def checkConnection (self, connectedLayer):
        # Check the current layer as active
        # Check the connected Nodes and set them as active
        self.active[connectedLayer[0]] = True 

        if connectedLayer[0] != 0:
            for i in range(2):
                connectedNode = connectedLayer[i + 2] # Get the connection node from the layer parameters
                self.checkConnection(self.decodedArch[connectedNode])

                if connectedLayer[1] is 'CON' or connectedLayer[1] is 'SUM': # If the layer is a concat or summation layer then both connections need to be checked
                    connectedNode2 = connectedLayer[3]
                    self.checkConnection(self.decodedArch[connectedNode2]) 