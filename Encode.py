from Architecture import Architecture as arch
from LayerDefinitions import LayerDefinitions as ld
import os

class Encode:

    def encode (architecture: arch, maxSize: int) -> list:
        encodedArch = []
        
        # Encode each layer of the architecture
        for i in range (maxSize + 1):
            print(i)
            layer = architecture.getLayer(i)
            encodedLayer = ld.encodeLayer(layer)
            if encodedLayer is not None:
                encodedArch.append(encodedLayer) # Encode the layer parameters and add them to the encoded architecture
        print(f"Encoded Architecture: {encodedArch}")

        return encodedArch