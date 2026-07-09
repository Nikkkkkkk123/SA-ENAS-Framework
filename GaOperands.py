from ArchitectureCodec import ArchitectureCodec as ac
from Architecture import Architecture as arch
import os
import random

class GaOperands:

    def __init__(self, crossoverRate, mutationRate):
        self.crossOverRate = crossoverRate
        self.mutationRate = mutationRate
    
    def crossover (self, parent1, parent2):
        
        if random.random() < self.crossOverRate:
            offspring1 = []
            offspring2 = []

            # Select two random points in the architecture to perform crossover
            point1 = random.randint(1, len(parent1.getFullArch()) - 1)
            point2 = random.randint(point1, len(parent1.getFullArch()) - 1)

            offspring1 = parent1.getFullArch()[0:point1] + parent2.getFullArch()[point1:point2] + parent1.getFullArch()[point2:]
            offspring2 = parent2.getFullArch()[0:point1] + parent1.getFullArch()[point1:point2] + parent2.getFullArch()[point2:]

            return offspring1, offspring2
        
        return parent1.getFullArch(), parent2.getFullArch()
    
    
    def mutation(self, architecture):
        """
            Mutation that may be performed on the architecture:
                - Add a layer (If maximum size has not been reached)
                - Remove a layer (if minimum size has not been reached)
                - Alter a layer (Currently im unsure if layers are treated as a single gene like layer type, connection, parameter or if check the layer as a whole and then decide what mutation occurs)
                - Layer alterations include:
                    - Change layer type
                    - change connections to the layer
                    - change the layer parameter
        """

        mutationDistributions = {
            "ADD": 0.1,
            "REMOVE": 0.1,
            "CHANGETYPE": 0.15,
            "CCONN": 0.2,
            "CPARAM": 0.40
        }

        candidateActive = len(ac.activeLayers(architecture))
        print(f"Candidate Active Layers: {candidateActive}")

        if candidateActive < 2:
            mutationDistributions["REMOVE"] = 0
        elif candidateActive >= len(architecture) - 1:
            mutationDistributions["ADD"] = 0
            
        for gene in architecture:
            if random.random() < 1:
                mutationChoice = random.choices(list(mutationDistributions.keys()), weights=list(mutationDistributions.values()))[0]
                self.mutationSwitch(architecture, mutationChoice, candidateActive)
        os._exit(0)
        return arch(architecture)
    
    def mutationSwitch (self, architecture, mutationChoice, lenActiveCand):
        if mutationChoice == "ADD" and lenActiveCand < len(architecture) - 1:
            print("Add Layer Mutation")
        elif mutationChoice == "REMOVE" and lenActiveCand > 1:
            print("Remove Layer Mutation")
        elif mutationChoice == "CHANGETYPE":
            print("Change Layer Type Mutation")
        elif mutationChoice == "CCONN":
            print("Change Layer Connection Mutation")
        elif mutationChoice == "CPARAM":
            print("Change Layer Parameter Mutation")