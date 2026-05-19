import random
from GenerateArchitecture import GenerateArchitecture as ga
from ArchitectureCodec import ArchitectureCodec as ac
"""
    NOTICE NOTICE NOTICE
    This is a notice that this file is currently not being used. It is currently being kept as a placeholder for an architecture class
    NOTICE NOTICE NOTICE
"""

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
        self.encoded = ac.encode(self.architecture)
        self.decoded = ac.decode(self.encoded)
        ac.activeLayers(self.architecture)
    

