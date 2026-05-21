from ArchitectureCodec import ArchitectureCodec as ac

class Architecture:

    """
    Function Name: __init__
    Description: Constructor for the Architecture class. This stores the encoded architecture, full architecture, and performance so that they can be easily obtained. 
    Parameter: 
        architecture: This is the full architecture that is going to be initialised.
    Return: 
        None
    """
    def __init__(self, architecture):
        self.fullArch = architecture
        self.encodedArch = ac.encode(architecture)
        self.activeArch = ac.activeLayers(architecture)
        self.performance = 0 # Currently placeholder
    
    """
    Below are getter functions for the architecture class.
    """
    def getEncodedArch(self):
        return self.encodedArch
    
    def getFullArch(self):
        return self.fullArch
    
    def getActiveArch(self):
        return self.activeArch
    
    def getFullArchLength(self):
        return len(self.fullArch)
    
    def getLayerIndex(self, layer):
        return self.fullArch.index(layer)
    
    def getConnectedLayer (self, layer, connection):
        if connection == 1:
            return self.fullArch[layer["Connection 1"]]
        elif connection == 2:
            return self.fullArch[layer["Connection 2"]]
        else:
            ValueError("Invalid connection number. Connection number must be 1 or 2.")


    

