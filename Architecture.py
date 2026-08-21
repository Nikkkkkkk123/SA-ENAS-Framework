import torch

from Node import Node as Node
from GenerateArchitecture import GenerateArchitecture as genArch
from LayerDefinitions import LayerDefinitions as ld
from Encode import Encode as encode
import os
import copy
import random
import numpy as np
class Architecture:

    _architecture: dict[int, Node] # Currently this is [int, int] but it should be node just hasnt been implemented yet
    _activeArchitecture: dict[int, Node]
    _layerSize: dict[int, int]
    _imageDimension: dict[int, int]
    _canRemove: bool
    _imageSize: int
    _fitness: float
    _noParameters: int
    _model: torch.nn.Module
    _trained: bool

    def __init__(self, maxLength: int, inputChannels: int, imageSize: int) -> None:
        self._layerSize = {}
        self._imageDimension = {}
        self._architecture = {}
        self._activeArchitecture = {}
        self._encodeActive = []
        self._encodedArchitecture = []

        self.maxSize = maxLength
        self.inputChannels = inputChannels
        self._imageSize = imageSize
        self._fitness = 0
        self._noParameters = 0
        self._model = None
        self._trained = False

    def generateArchitecture (self) -> None:
        newNode: Node = None
        for i in range(0, self.maxSize):
            newNode = genArch().generateLayer(self._architecture, i, self.inputChannels, self._imageSize)
            self.addNode(newNode.getNodeId(), newNode)

        finalNode = genArch().generateOutputLayer(self._architecture, self.maxSize)
        self.addNode(finalNode.getNodeId(), finalNode)
        self.getActive()
        self._encodedArchitecture = encode.encode(list(self._architecture.values()), self.maxSize)


    def buildArchitecture (self, newArchitecture: dict[int, Node]) -> bool:
        self._architecture = {}
        self._activeArchitecture = {}
        for node in newArchitecture.values():
            node.setActive(False)
            if genArch().buildLayer(node):
                self.addNode(node.getNodeId(), node)
            else:
                return False
        self.getActive()
        self._encodedArchitecture = encode.encode(list(self._architecture.values()), self.maxSize)
        return True

    def setArchitecture (self, newArchitecture: list[Node]) -> bool:
        self.addNode(newArchitecture[0].getNodeId(), newArchitecture[0])
        for node in newArchitecture[1:]:
            newNode = copy.deepcopy(node)
            newNode.setActive(False)
            newNode.changeConnection1(self._architecture.get(newNode.getConnection1().getNodeId()))

            if newNode.getConnection2() is not None:
                newNode.changeConnection2(self._architecture.get(newNode.getConnection2().getNodeId()))
            if newNode.getNodeId() != 0 and genArch().validateLayer(newNode) is False:
                return False
            
            self.addNode(newNode.getNodeId(), newNode)
        self.getActive()
        self._encodedArchitecture = encode.encode(list(self._architecture.values()), self.maxSize)
        return True

    def getLayerSize (self, layerId: int) -> int:
        return self._layerSize.get(layerId, 0)

    def getLayer (self, layerId: int) -> Node:
        return self._architecture.get(layerId, None)

    def getImageDimension (self, layerId: int) -> int:
        return self._imageDimension.get(layerId, 0)

    def addNode (self, key: int, newNode: Node) -> bool:
        if self._containsKey(key):
            return False

        self._architecture[key] = newNode
        return True

    def getActive(self) -> dict[int, Node]:
        connectionSet = set()
        if self.checkConnections(self._architecture.get(self.maxSize), connectionSet):
            self._encodeActive = self.findActiveEncoding()
        return self._activeArchitecture

    def checkConnections (self, node: Node, connectionSet: set) -> bool:
        if node is None or node.getNodeId() in connectionSet:
            return False

        node.setActive(True)
        newActiveNode = copy.deepcopy(node)
        if node.getNodeType() == ld.getInputLayerStr():
            self._activeArchitecture[node.getNodeId()] = newActiveNode
            return True
        connectionSet.add(node.getNodeId())

        self.checkConnections(node.getConnection1(), connectionSet)
        if node.requiresTwoConnections():
            self.checkConnections(node.getConnection2(), connectionSet)

        if self._activeArchitecture.get(node.getNodeId()) is None:
            self._activeArchitecture[node.getNodeId()] = newActiveNode
            newActiveNode._connection1 = self._activeArchitecture.get(node.getConnection1().getNodeId())
            if node.requiresTwoConnections():
                newActiveNode._connection2 = self._activeArchitecture.get(node.getConnection2().getNodeId())
        else:
            self._activeArchitecture.pop(node.getNodeId())
            self._activeArchitecture[node.getNodeId()] = newActiveNode

        newActiveNode._nodeId = list(self._activeArchitecture.keys()).index(node.getNodeId())
        return True

    def _containsKey (self, key: int) -> bool:
        if self._architecture.get(key) is None:
            return False
        return True

    def getNode (self, key: int) -> Node:
        if not self._containsKey(key):
            return None
        return self._architecture[key]

    # This is not currently not going to be used but it is my idea to avoid training duplicate architectures.
    # But a encoding class will be made which will return the encoded version which then will be stored to avoid duplicates
    def findActiveEncoding (self):
        encoding = []
        encoding = encode.encode(list(self._activeArchitecture.values()), len(self._activeArchitecture) - 1)
        return encoding

    def getActiveArchLength (self) -> int:
        return len(self._activeArchitecture)   

    def getActiveArch (self) -> dict[int, Node]:
        return self._activeArchitecture

    def canAdd (self) -> bool:
        if self.getActiveArchLength() < self.maxSize - 1:
            return True
        return False

    def canRemove (self) -> bool:
        if self.getActiveArchLength() > 2:
            return True
        return False

    """
    Below are the functions regarding toggling a layer to either be active or inactive.
    These functions check if the layer being selected is active or inactive and then appends the architecture accordingly
    """
    def toggleLayer (self, togLayer: Node) -> bool:
        if togLayer is None:
            raise ValueError(f"Node with ID {togLayer.getNodeId()} does not exist in the architecture")

        if ld.canMutateOption(togLayer.getNodeType(), "TOG"):
            if togLayer.isActive():
                return self._toggleRemoveLayer(togLayer.getNodeId())
            else:
                return self._toggleAddLayer(togLayer.getNodeId())

        return False
        

    def _toggleRemoveLayer (self, layerKey: int) -> bool:
        if self._activeArchitecture.get(layerKey) is None:
            raise ValueError(f"Node with ID {layerKey} is not currently apart of the active architecture")

        if not self.canRemove():
            return False

        # any node pointing to it in the active architecture needs to now point to the removing nodes input connection
        # I decided to only take connection 1 since its the primary input. if it was a node like sum, if that mutated to anything else
        # it would only take connection 1. 
        for key in list(self._activeArchitecture.keys()):
            if self._architecture[key].getConnection1() == self.getNode(layerKey):
                self._architecture[key].changeConnection1(self.getNode(layerKey).getConnection1())
            if self._architecture[key].getConnection2() == self.getNode(layerKey):
                self._architecture[key].changeConnection2(self.getNode(layerKey).getConnection1())

        self._removeFromActiveArchitecture(layerKey)
        return True

    def _removeFromActiveArchitecture (self, nodeID: int) -> None:
        # This is just in case. but before this function it should be checked
        if self._activeArchitecture.get(nodeID) is None:
            raise ValueError(f"Node with ID {nodeID} is not currently apart of the active architecture")
        self._activeArchitecture.pop(nodeID)

    def _toggleAddLayer (self, layerKey: int) -> bool:
        # Just in case this should be checked before this function is called
        if self._activeArchitecture.get(layerKey) is not None:
            raise ValueError(f"Node with ID {layerKey} is already apart of the active architecture")

        if not self.canAdd():
            return False
        
        # My current plan is to find the first active layer after this node then make its connection 1 point to this and then make this node take the original as its incoming
        # then find the next active layer taking that node as an input and swap it
        # [input -> a -> b -> output]
        #           | -> c           
        # become
        # [input -> a -> c -> b -> output]
        for key in list(self._activeArchitecture.keys()):
            if key > layerKey:
                self._architecture[layerKey].changeConnection1(self._architecture[key].getConnection1())
                self._architecture[key].changeConnection1(self.getNode(layerKey))
                break

        return True

    """
    Below are functions regarding mutating the connections of a layer
    """
    def mutateConnection (self, layer: Node) -> bool:
        layerKey = layer.getNodeId()
        if self._architecture.get(layerKey) is None:
            raise ValueError(f"Node with ID {layerKey} does not exist in the architecture")

        # If the layer requires two connections then randomly pick either one
        newNodeId = random.randint(0, layerKey - 1)
        newConnection = self._architecture.get(newNodeId)

        if layer.requiresTwoConnections():
            return self._mutateTwoConnections(layer, newConnection)

        layer.changeConnection1(newConnection)
        return True

    def _mutateTwoConnections (self, layer: Node, newConnection: Node) -> bool:
        layerKey = layer.getNodeId()
        if self._architecture.get(layerKey) is None:
            raise ValueError(f"Node with ID {layerKey} does not exist in the architecture")

        connectionChoice = random.choice([1, 2])

        if connectionChoice == 1:
            self._architecture[layerKey].changeConnection1(newConnection)
        else:
            self._architecture[layerKey].changeConnection2(newConnection)
        return True

    """
    Below are the functions regarding mutating the parameters of a layer
    """

    def mutateParameters (self, layer: Node) -> bool:
        layerKey = layer.getNodeId()
        if self._architecture.get(layerKey) is None:
            raise ValueError(f"Node with ID {layerKey} does not exist in the architecture")

        selectedParameter: str
        selectedParameter = ld.selectParameterMutation(layer.getNodeType())

        if selectedParameter == None:
            return False

        return self._mutateSelectedParam(layer, selectedParameter)

    def _mutateSelectedParam (self, layer: Node, selectedParameter: str) -> bool:

        match selectedParameter:
            case "filterSize":
                newFilter = ld.selectFilterSize(layer.getNodeType())
                self._architecture[layer.getNodeId()].setFilterSize(newFilter)
                return True
            case "kernelSize":
                newKernel = ld.selectKernelSize(layer.getNodeType())
                self._architecture[layer.getNodeId()].setKernelSize(newKernel)
                return True
            
        return False

    """
    Below are functions regarding changing the layer type
    """

    def changeLayerType (self, layer: Node) -> bool:
        layerKey = layer.getNodeId()
        if self._architecture.get(layerKey) is None:
            raise ValueError(f"Node with ID {layerKey} does not exist in the architecture")

        newLayerType = ld.selectNewLayerType(layer.getNodeType())

        if newLayerType is None:
            return False

        # Check that the parameters match it
        self._checkValidFilter(layer, newLayerType)
        self._checkValidKernel(layer, newLayerType)

        self._architecture[layerKey].setNodeType(newLayerType)
        return True

    def _checkValidFilter (self, layer: Node, newLayerType: str) -> None:
        if not ld.checkValidFilter(newLayerType, layer.getFilterSize()):
            newFilter = ld.selectFilterSize(newLayerType)
            self._architecture[layer.getNodeId()].setFilterSize(newFilter)

    def _checkValidKernel (self, layer: Node, newLayerType: str) -> None:
        if not ld.checkValidKernel(newLayerType, layer.getKernelSize()):
            newKernel = ld.selectKernelSize(newLayerType)
            self._architecture[layer.getNodeId()].setKernelSize(newKernel)

    def print (self) -> None:
        for layer in self._activeArchitecture.values():
            print(f"Layer {layer.getNodeId()} is of type {layer.getNodeType()} with connection 1 being {layer.getConnection1().getNodeId() if layer.getConnection1() is not None else None} and connection 2 being {layer.getConnection2().getNodeId() if layer.getConnection2() is not None else None}")

    def getActiveEncoding (self) -> list:
        if len(self._encodeActive) == 0:
            print("Active encoding is None, generating active encoding")
            os._exit(0)
        return self._encodeActive

    def getFitness (self) -> float:
        return self._fitness

    def calculateFitness (self, f1Score: float) -> None:
        self._fitness =  f1Score - (self._noParameters / 1000000)

    def getNoParameters (self) -> int:
        return self._noParameters

    def setNoParameters (self, newNoParameters: int) -> None:
        self._noParameters = newNoParameters

    def setModel (self, model: torch.nn.Module) -> None:
        self._model = model

    def getModel (self) -> torch.nn.Module:
        return self._model

    def getEncodedArchitecture (self) -> list:
        return [layer for layers in self._encodedArchitecture for layer in layers]

    def predictFitness (self, surrogateModel) -> None:
        self._fitness = surrogateModel.predict(np.asarray([self.getEncodedArchitecture()]).reshape(1, -1))[0]
        return self._fitness

    def setTrained (self, trained: bool) -> None:
        self._trained = trained

    def getTrained (self) -> bool:
        return self._trained
    