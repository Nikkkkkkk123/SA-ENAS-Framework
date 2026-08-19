from __future__ import annotations
import os
class Node:
    _nodeId: int
    _nodeType: str
    _connection1: Node
    _connection2: Node
    _isActive: bool
    _filterSize: int
    _kernelSize: int
    _imageDimension: int
    _layerSize: int

    def __init__ (self, key: int, nodeType: str, connection1: Node, connection2: Node = None, filterSize: int = None, kernelSize: int = None, layerSize: int = None, imageDimension: int = None) -> None:
        self._nodeId = key
        self._nodeType = nodeType
        self._connection1 = connection1 # this means input
        self._connection2 = connection2
        self._isActive = False
        self._filterSize = filterSize
        self._kernelSize = kernelSize
        self._imageDimension = imageDimension
        self._layerSize = layerSize

    @classmethod
    def createInputNode (cls) -> Node:
        return cls(0, "input", None, None, 0, 0)

    def getNodeType (self) -> str:
        return self._nodeType

    def setNodeType (self, newType: str) -> None:
        self._nodeType = newType

    def getNodeId (self) -> int:
        return self._nodeId

    def getConnection1 (self) -> Node:
        return self._connection1

    def getConnection2 (self) -> Node:
        return self._connection2
    
    def setActive (self, isActive: bool) -> None:
        self._isActive = isActive

    def isActive (self) -> bool:
        return self._isActive

    def requiresTwoConnections (self) -> bool:
        if self.getNodeType() == "SUM" or self.getNodeType() == "CON":
            return True
        return False
    
    def changeConnection1 (self, newConnection: Node) -> None:
        self._connection1 = newConnection

    def changeConnection2 (self, newConnection: Node) -> None:
        self._connection2 = newConnection

    def getFilterSize (self) -> int:
        return self._filterSize

    def getKernelSize (self) -> int:
        return self._kernelSize

    def setFilterSize (self, filterSize: int) -> None:
        self._filterSize = filterSize
    def setKernelSize (self, kernelSize: int) -> None:
        self._kernelSize = kernelSize

    def getImageDimension (self) -> int:
        return self._imageDimension
    def setImageDimension (self, dimension: int) -> None:
        self._imageDimension = dimension

    def setLayerSize (self, size: int) -> None:
        self._layerSize = size
    def getLayerSize (self) -> int:
        return self._layerSize

    def setFitness (self, fitness: int) -> None:
        self._fitness = fitness

    def getFitness (self) -> int:
        return self._fitness

    def getConnectionOutputSize (self, connectionNumber: int) -> int:
        if connectionNumber == 1:
            return self.getConnection1().getLayerSize()
        elif connectionNumber == 2:
            return self.getConnection2().getLayerSize()
        else:
            raise ValueError("Invalid connection number. Must be 1 or 2.")

    def getConnectionImageDimension (self, connectionNumber: int) -> int:
        if connectionNumber == 1:
            return self._connection1.getImageDimension() if self._connection1 is not None else 0
        elif connectionNumber == 2:
            return self._connection2.getImageDimension() if self._connection2 is not None else 0
        else:
            raise ValueError("Invalid connection number. Must be 1 or 2.")

    def getNodeInformation (self):
        nodeInfo = []

        nodeInfo.append(self._nodeType)

        if self._connection1 is not None:
            nodeInfo.append(self._connection1.getNodeId())

        if self._connection2 is not None:
            nodeInfo.append(self._connection2.getNodeId())

        if self._filterSize is not None:
            nodeInfo.append(self._filterSize)

        if self._kernelSize is not None:
            nodeInfo.append(self._kernelSize)

        return nodeInfo

    def print(self) -> None:
        print(f"Node ID: {self._nodeId}, Node Type: {self._nodeType}, Connection1: {self._connection1.getNodeId() if self._connection1 is not None else 'None'}, Connection2: {self._connection2.getNodeId() if self._connection2 is not None else 'None'}, Filter Size: {self._filterSize}, Kernel Size: {self._kernelSize}, Is Active: {self._isActive}")