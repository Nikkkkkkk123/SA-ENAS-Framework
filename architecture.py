import layers
import random

class Architecture:

    def __init__(self, length):
        self.length = length
        self.architecture = layers.Layer(length)
        active, arch = self.architecture.generate_architecture()