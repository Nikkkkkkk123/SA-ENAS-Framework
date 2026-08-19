from LayerDefinitions import LayerDefinitions as ld
import os

class Encode:

    def encode (architecture: list, maxSize: int) -> list:
        encodedArch = []
        
        # Encode each layer of the architecture
        for i in range (maxSize + 1):
            layer = architecture[i]
            encodedLayer = ld.encodeLayer(layer)
            if encodedLayer is not None:
                encodedArch.append(encodedLayer) # Encode the layer parameters and add them to the encoded architecture

        return encodedArch