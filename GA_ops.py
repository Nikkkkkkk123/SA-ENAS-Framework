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

        parent1 = GA_ops._selectParent(currPopulation)
        parent2 = GA_ops._selectParent(currPopulation)

        return parent1, parent2

    # Helper function
    def _selectParent (currPopulation: list[arch]) -> arch:
        selectionPopulation = random.sample(currPopulation, 3) if len(currPopulation) >= 3 else random.sample(currPopulation, len(currPopulation))
        fittestCandidate = max(selectionPopulation, key=lambda candidate: candidate.getFitness())
        return fittestCandidate

    def performGA (currPopulation: list[arch], maxSize: int, inputChannels: int, imageSize: int, mutationRate: float, crossoverRate: float) -> tuple[arch, arch]:
        parent1: arch = None
        parent2: arch = None

        offspringArch_1: arch = None
        offspringArch_2: arch = None

        # This loop is really only there so that new parents can be selected if max crossover attempts has been reached. This should not infinitetly loop
        while offspringArch_1 is None or offspringArch_2 is None:
            parent1, parent2 = GA_ops.selectParents(currPopulation)

            offspringArch_1, offspringArch_2 = GA_ops._crossover(parent1, parent2, maxSize, inputChannels, imageSize, crossoverRate)

            # Mutation attempts 10 times, it it fails to mutate it will return nothing and loop to try again with new parents. If it is none then there is no use mutating the second offspring just have to try again
            offspringArch_1 = GA_ops.mutation(offspringArch_1, maxSize, mutationRate)

            if offspringArch_1 is not None:
                offspringArch_2 = GA_ops.mutation(offspringArch_2, maxSize, mutationRate)

        return offspringArch_1, offspringArch_2

    def _crossover (parent1: arch, parent2: arch, maxSize: int, inputChannels: int, imageSize: int, crossoverRate: float) -> tuple[arch, arch]:
        crossoverChance = random.random()
        # Check if crossover will occur. if it does then perform 2 point crossover. Otherwise return the parents as the offspring
        if crossoverChance <= crossoverRate:
            for i in range (1, 10):
                point1 = random.randint(1, maxSize - 2)
                point2 = random.randint(point1 + 1, maxSize - 1)

                offspring1 = list(parent1._architecture.values())[0:point1] + list(parent2._architecture.values())[point1:point2] + list(parent1._architecture.values())[point2:]
                offspring2 = list(parent2._architecture.values())[0:point1] + list(parent1._architecture.values())[point1:point2] + list(parent2._architecture.values())[point2:]

                offspringArch_1 = arch(maxSize, inputChannels, imageSize)
                offspringArch_2 = arch(maxSize, inputChannels, imageSize)

                if offspringArch_1.setArchitecture(offspring1) and offspringArch_2.setArchitecture(offspring2):
                    return offspringArch_1, offspringArch_2
        else:
            return arch(maxSize, inputChannels, imageSize).setArchitecture(list(parent1._architecture.values())), arch(maxSize, inputChannels, imageSize).setArchitecture(list(parent2._architecture.values()))
        return None, None

    """
    Below are functions regarding performing mutation on architectures
    """
    def mutation (offspring: arch, maxSize: int, mutationRate: float) -> arch:
        # A check incase the crossover failed. If it did then return None to try again. This would mean crossover reached the maximum attempts with those parents and to try again
        if offspring is None:
            return None
        
        mutateType: str = None
        for _ in range (1, 10):
            offSpringCopy: arch = copy.deepcopy(offspring) # This copy is used to store the original offspring incase mutation was unsucessful
            for i in range (1, maxSize):
                mutateChance = random.random()
                if mutateChance < mutationRate:
                    succMutate: bool = False
                    while not succMutate:
                        mutationType = random.choice(GA_ops.mutationOptions)
                        succMutate = GA_ops.mutateSwitch(offSpringCopy, offSpringCopy.getLayer(i), mutationType)

            # Check the mutation is valid
            # The check is inside the for loop so it can return if the architecture is valid. 
            if offSpringCopy.buildArchitecture(offSpringCopy._architecture):
                return offSpringCopy
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
