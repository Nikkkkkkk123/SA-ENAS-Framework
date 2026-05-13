import random
from GenerateArchitecture import GenerateArchitecture as ga
from ArchitectureCodec import ArchitectureCodec as ac

class Architecture:
    """
    This is to be changed. Currently it is an architecture class that calls the generate architecture functions. While it will still do this
    it is going to be an object class. so it will store the architecture information including decoded and encoded versions.
    Additionally, it will store the active layers to avoid having to check each time and when actual training of CNN architectures is implemented
    it will also store the fitness of the architecture
    """

    def __init__(self, length):
        self.length = length
        self.architecture = ga.generateArchitecture(length)
        self.encoded = self.architecture # Still have to implement the encoding function
        ac.encode(self.architecture)
    

