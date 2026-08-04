from Node import Node
from LayerDefinitions import LayerDefinitions as ld
import random

class GenerateArchitecture:

    def generateLayer (self, architecture: dict[int, Node], nodeId: int, inputChannels: int, imageSize: int) -> Node:
        newNode: Node = None

        if nodeId == 0:
            return GenerateArchitecture.generateInputLayer(nodeId, 0, inputChannels, imageSize)

        while newNode is None:
            layerType = ld.selectNewLayerType()

            connection1 = architecture[random.randint(0, nodeId - 1)]
            connection2 = architecture[random.randint(0, nodeId - 1)]

            filterSize = ld.selectFilterSize(layerType)
            kernelSize = ld.selectKernelSize(layerType)

            # The new image dimension, and layer size
            newNode = Node(nodeId, layerType, connection1, connection2, filterSize, kernelSize, inputChannels, imageSize)

            # This is needed to ensure its not just straight pooling to where dimensions become 0
            if self.layerSwitch(newNode) is False:
                newNode = None
        return newNode

    def buildLayer (self, layer: Node) -> bool:
        if layer.getNodeId() == 0:
            return GenerateArchitecture.generateInputLayer(layer.getNodeId(), 0, layer.getLayerSize(), layer.getImageDimension())

        if self.layerSwitch(layer) is False:
            return False
        return True

    def validateLayer (self, layer: Node) -> bool:
        if self.layerSwitch(layer) is False:
            return False
        return True

    def generateInputLayer (self, nodeId: int, inputChannels: int, imageSize: int) -> Node:
        return Node(nodeId, "IN", None, None, None, None, inputChannels, imageSize)

    def generateOutputLayer (self, architecture: dict[int, Node], nodeId: int) -> Node:
        connection1Id = random.randint(0, nodeId - 1)
        connection1 = architecture[connection1Id]
        newNode = Node(nodeId, "LIN", connection1, None, None, None, None, None)
        newNode._layerSize = connection1._layerSize
        newNode._imageDimension = connection1._imageDimension
        return newNode

    # This is done to calculate the output dimensions of the layers and images to ensure valid layers are being put in and avoid architectures that may just be pooling to image dimensions are 0
    def layerSwitch (self, layer: Node) -> bool:
        connectionSize = layer.getConnection1()._layerSize
        match layer.getNodeType():
            case "CB":
                newImageDimension = int (((layer.getConnectionImageDimension(1) - layer.getKernelSize() + (2 * (layer.getKernelSize() // 2))) / 1) + 1)

                if self.imageDimensionCheck(newImageDimension):
                    layer._layerSize = layer._filterSize
                    layer._imageDimension = newImageDimension
                    return True
                return False
            case "RB":
                newImageDimension = int (((layer.getConnectionImageDimension(1) - layer.getKernelSize() + (2 * (layer.getKernelSize() // 2))) / 1) + 1)

                if self.imageDimensionCheck(newImageDimension):
                    layer._layerSize = max(connectionSize, layer._filterSize)
                    layer._imageDimension = newImageDimension

                    return True
                return False
            case "SUM":
                newImageDimension = min(layer.getConnectionImageDimension(1), layer.getConnectionImageDimension(2))

                if self.imageDimensionCheck(newImageDimension):
                    layer._imageDimension = newImageDimension
                    layer._layerSize = max(connectionSize, layer.getConnectionOutputSize(2))

                    return True
                return False
            case "CON":
                newImageDimension = min(layer.getConnectionImageDimension(1), layer.getConnectionImageDimension(2))

                if self.imageDimensionCheck(newImageDimension):
                    layer._imageDimension = newImageDimension
                    layer._layerSize = connectionSize + layer.getConnectionOutputSize(2)

                    return True

                return False
            case "MP" | "AP":
                newImageDimension = int(((layer.getConnectionImageDimension(1) - 2) / 2) + 1)

                if self.imageDimensionCheck(newImageDimension):
                    layer._imageDimension = newImageDimension
                    layer._layerSize = connectionSize

                    return True
                return False
            case "LIN":
                layer._imageDimension = layer.getConnectionImageDimension(1)
                layer._layerSize = layer.getConnectionOutputSize(1)
                return True

    def getConnectionSize (self, layer: Node, connectionNumber: int) -> int:
        if connectionNumber == 1:
            return layer.getConnectionOutputSize(1)
        elif connectionNumber == 2:
            return layer.getConnectionOutputSize(2)
        else:
            raise ValueError("Invalid connection number. Must be 1 or 2.")

    def imageDimensionCheck (self, imageDimension: int) -> bool:
        if imageDimension < 1:
            return False
        return True
