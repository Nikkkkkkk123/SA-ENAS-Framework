import random
from xml.dom import Node
from kernelSize import KERNEL_SIZE
from FilterSize import FILTER_SIZE

class LayerDefinitions:
    LAYERINDEX: dict[int, str] = {
        0: "IN",
        1: "RB",
        2: "CB",
        3: "MP",
        4: "AP",
        5: "CON",
        6: "SUM",
        7: "LIN"
    }

    INDEXLAYER: dict[str, int] = {
        "IN": 0,
        "RB": 1,
        "CB": 2,
        "MP": 3,
        "AP": 4,
        "CON": 5,
        "SUM": 6,
        "LIN": 7
    }

    SELECTABLELAYERS: set[str] = {"RB", "CB", "MP", "AP", "CON", "SUM"}

    MUTATIONRULES: dict[str, list[str]] = {
        "IN": [],
        "RB": ["TOG", "CHANGETYPE", "CCONN"],
        "CB": ["TOG", "CHANGETYPE", "CCONN"],
        "MP": ["TOG", "CHANGETYPE", "CCONN"],
        "AP": ["TOG", "CHANGETYPE", "CCONN"],
        "CON": ["TOG", "CHANGETYPE", "CCONN"],
        "SUM": ["TOG", "CHANGETYPE", "CCONN"],
        "LIN": ["CHANGETYPE"]
    }

    LAYERPARAMETERS: dict[str, list[str]] = {
        "IN": [],
        "RB": ["filterSize", "kernelSize"],
        "CB": ["filterSize", "kernelSize"],
        "MP": ["kernelSize"],
        "AP": ["kernelSize"],
        "CON": [],
        "SUM": [],
        "LIN": []
    }

    @classmethod
    def selectNewLayerType (cls, current: str = None) -> str:
        if current is not None:
            if LayerDefinitions.canMutateOption(current, "CHANGETYPE"):
                return random.choice((list(LayerDefinitions.SELECTABLELAYERS - {current})))
            return None
        return random.choice(list(LayerDefinitions.SELECTABLELAYERS))

    @classmethod
    def getInputLayerStr (cls) -> str:
        return LayerDefinitions.LAYERINDEX[LayerDefinitions.INDEXLAYER["IN"]]

    @classmethod
    def getOutputLayerStr (cls) -> str:
        return LayerDefinitions.LAYERINDEX[LayerDefinitions.INDEXLAYER["LIN"]]

    @classmethod
    def canMutateOption (cls, layerType: str, mutationType: str) -> bool:
        if mutationType in LayerDefinitions.MUTATIONRULES[layerType]:
            return True
        return False

    @classmethod
    def getLayerIndex (cls, layerType: str) -> int:
        return LayerDefinitions.INDEXLAYER[layerType]

    """
    Helper functions to select kernel and filter sizes for layers
    """
    @classmethod
    def selectParameterMutation (cls, layerType: str) -> str:
        choices: list[str] = LayerDefinitions.LAYERPARAMETERS[layerType]
        if len(choices) > 0:
            return random.choice(choices)
        return None
    
    @classmethod
    def selectFilterSize (cls, current: str) -> int:
        if "filterSize" in LayerDefinitions.LAYERPARAMETERS[current]:
            return random.choice((list(FILTER_SIZE - {current})))
        return 0

    @classmethod
    def selectKernelSize (cls, current: str) -> int:
        if "kernelSize" in LayerDefinitions.LAYERPARAMETERS[current]:
            return random.choice((list(KERNEL_SIZE - {current})))
        return 0

    # Need to check that valid parameters are bing used for a specific layertype
    @classmethod
    def checkValidFilter (cls, layerType: str, filterSize: int) -> bool:
        if "filterSize" in LayerDefinitions.LAYERPARAMETERS[layerType]:
            if filterSize in FILTER_SIZE:
                return True
        return False

    @classmethod
    def checkValidKernel (cls, layerType: str, kernelSize: int) -> bool:
        if "kernelSize" in LayerDefinitions.LAYERPARAMETERS[layerType]:
            if kernelSize in KERNEL_SIZE:
                return True
        return False

    @classmethod
    def encodeLayer (cls, layer: Node) -> list:
        layerType = layer.getNodeType()
        encodedType = LayerDefinitions.getLayerIndex(layerType)
        if LayerDefinitions.LAYERINDEX[encodedType] == "IN":
            return None
        connection1 = layer.getConnection1().getNodeId() if layer.getConnection1() is not None else 0
        connection2 = layer.getConnection2().getNodeId() if layer.getConnection2() is not None else 0
        encodedFilterSize = 0 if layer.getFilterSize() is None or layer.getFilterSize() == 0 else list(FILTER_SIZE).index(layer.getFilterSize())
        encodedKernelSize = 0 if layer.getKernelSize() is None or layer.getKernelSize() == 0 else list(KERNEL_SIZE).index(layer.getKernelSize())

        if LayerDefinitions.LAYERINDEX[encodedType] == "LIN":
            return [encodedType, connection1]
        return [encodedType, connection1, connection2, encodedFilterSize, encodedKernelSize]