import random

class GenerateLayers:
    
    kernalSize = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    filterSize = [8, 16, 32, 64, 128, 256, 512]

    @staticmethod
    def generatePoolLayer (type, layerIndex):
        poolLayer = {
            "type": "MP" if type == 3 else "AP",
            "Connection 1": random.randint(0, layerIndex),
            "Connection 2": random.randint(0, layerIndex),
            "Filter Size": 0,
            "Kernel Size": random.choice(GenerateLayers.kernalSize)
        }

        return poolLayer
    
    @staticmethod
    def generateConvLayer (layerIndex):
        convLayer = {
            "type": "CB",
            "Connection 1": random.randint(0, layerIndex),
            "Connection 2": random.randint(0, layerIndex),
            "Filter Size": random.choice(GenerateLayers.filterSize),
            "Kernel Size": random.choice(GenerateLayers.kernalSize)
        }

        return convLayer
    
    @staticmethod
    def generateResBlock (layerIndex):
        resBlock = {
            "type": "RB",
            "Connection 1": random.randint(0, layerIndex),
            "Connection 2": random.randint(0, layerIndex),
            "Filter Size": random.choice(GenerateLayers.filterSize),
            "Kernel Size": random.choice(GenerateLayers.kernalSize)
        }

        return resBlock
    
    @staticmethod
    def generateBottleNeckDepthWise (layerIndex):
        bottleNeckDepthWise = {
            "type": "BND",
            "Connection 1": random.randint(0, layerIndex),
            "Connection 2": random.randint(0, layerIndex),
            "Filter Size": random.choice(GenerateLayers.filterSize),
            "Kernel Size": random.choice(GenerateLayers.kernalSize)
        }

        return bottleNeckDepthWise
    
    @staticmethod
    def generateCombineBlock (type, layerIndex):
        combineBlock = {
            "type": "CON" if type == 6 else "SUM",
            "Connection 1": random.randint(0, layerIndex),
            "Connection 2": random.randint(0, layerIndex),
            "Filter Size": 0,
            "Kernel Size": 0
        }

        # Below is to check that the two connected layers are not the same
        # If they are then two different connection nodes are randomly selected
        # The first is then checked. if same then uses the second number
        if combineBlock["Connection 1"] == combineBlock["Connection 2"]:
            potentialConnections = random.sample(range(0, layerIndex), 2)

            if combineBlock["Connection 1"] == potentialConnections[0]:
                combineBlock["Connection 2"] = potentialConnections[1]
            else:
                combineBlock["Connection 2"] = potentialConnections[0]

        return combineBlock
    
    @staticmethod
    def generateOutputLayer (layerIndex):
        outputLayer = {
            "type": "OUT",
            "Connection 1": random.randint(1, layerIndex), # Currently enforcing not picking the input layer. At the moment this is a test input but may become rule
        }

        return outputLayer
    
    @staticmethod
    def generateLayer (blockID, layerIndex):
        # This is just for testing purposes to generate specific layer types for testing
        match blockID:
            case 1:
                return GenerateLayers.generateResBlock(layerIndex)
            case 2:
                return GenerateLayers.generateConvLayer(layerIndex)
            case 3:
                return GenerateLayers.generatePoolLayer(3, layerIndex)
            case 4:
                return GenerateLayers.generatePoolLayer(4, layerIndex)
            case 5:
                blockID = random.randint(1, 2)
                return GenerateLayers.generateLayer(blockID, layerIndex)
                #return GenerateLayers.generateBottleNeckDepthWise(layerIndex)
            case 6 | 7:
                # Ensure that the combining layers can actually be used
                # If they cant then just select a new layer type
                if layerIndex < 2:
                    blockID = random.randint(1, 5) # Currently a magic number
                    return GenerateLayers.generateLayer(blockID, layerIndex) 
                return GenerateLayers.generateCombineBlock(type=blockID, layerIndex=layerIndex)
            case 8:
                return GenerateLayers.generateOutputLayer(layerIndex)
