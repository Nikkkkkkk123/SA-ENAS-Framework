from Architecture import Architecture as arch
from Node import Node

import copy
import random
import os

class GA_ops:
    mutationOptions = ["TOG", "CHANGETYPE", "CCONN", "CPARAM"]

    def selectParents (currPopulation: list[arch]) -> tuple[arch, arch]:
        parent1: arch = None
        parent2: arch = None

        selectionPopulation = random.sample(currPopulation, 3)
        parent1 = random.choice(selectionPopulation)
        selectionPopulation.remove(parent1)
        parent2 = random.choice(selectionPopulation)

        return parent1, parent2

    def crossover (parent1: arch, parent2: arch, maxSize: int, inputChannels: int, imageSize: int) -> tuple[arch, arch]:
        offspringArch_1: arch = None
        offspringArch_2: arch = None

        while offspringArch_1 is None or offspringArch_2 is None:  # perform crossover
            point1 = random.randint(1, maxSize - 2)
            point2 = random.randint(point1 + 1, maxSize - 1)

            offspring1 = list(parent1._architecture.values())[0:point1] + list(parent2._architecture.values())[point1:point2] + list(parent1._architecture.values())[point2:]
            offspring2 = list(parent2._architecture.values())[0:point1] + list(parent1._architecture.values())[point1:point2] + list(parent2._architecture.values())[point2:]

            offspringArch_1 = arch(maxSize, inputChannels, imageSize)
            offspringArch_2 = arch(maxSize, inputChannels, imageSize)

            if offspringArch_1.setArchitecture(offspring1) and offspringArch_2.setArchitecture(offspring2):
                offspringArch_1 = GA_ops.mutation(offspringArch_1, maxSize)
                offspringArch_2 = GA_ops.mutation(offspringArch_2, maxSize)
            else:
                offspringArch_1 = None
                offspringArch_2 = None
        return offspringArch_1, offspringArch_2

    """
    Below are functions regarding performing mutation on architectures
    """
    def mutation (offspring: arch, maxSize: int) -> arch:
        mutateType: str = None
        for i in range (1, maxSize):
            succMutate: bool = False

            while not succMutate:
                mutationType = random.choice(GA_ops.mutationOptions)
                succMutate = GA_ops.mutateSwitch(offspring, offspring.getLayer(i), mutationType)

        # Check the mutation is valid
        if offspring.buildArchitecture(offspring._architecture):
            return offspring
        print("Mutation failed, returning original architecture")
        return None

    def mutateSwitch (architecture: arch, layer: Node, mutationType: str) -> bool:
        match mutationType:
            case "TOG":
                if architecture.toggleLayer(layer):
                    return True
                return False
            case "CHANGETYPE":
                if architecture.changeLayerType(layer):
                    return True
                return False
            case "CCONN":
                if architecture.mutateConnection(layer):
                    return True
                return False
            case "CPARAM":
                if architecture.mutateParameters(layer):
                    return True
                return False
