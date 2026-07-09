"""
File Name: Evolve.py
Description: This file is the main evolutionary file containing the main loop.
"""
#import rename
from GenerateArchitecture import GenerateArchitecture as ga
from ArchitectureCodec import ArchitectureCodec as ac
from Architecture import Architecture as arch
from GaOperands import GaOperands as gaOperands
import LayerBlocks
import copy
import numpy as np
import torch
from tqdm import tqdm
import os
import random
from torchinfo import summary

class Evolve:

    """ 
    Function Name: __init__
    Description: This is the constructor function for the evolution file
    Parameter: 
        populationSize: The number of architectures to generate
        generations: The number of generations to evolve
        maxSize: The maximum length of the architecture to generate (including the input/output layer)
        imageColor: The number of channels for the input image (RGB = 3, Grayscale = 1)
        crossOverRate: The probability of crossover occurring
    Return: 
        None
    """
    def __init__(self, populationSize, generations, maxSize, imageColor, crossOverRate, mutationRate):
        self.populationSize = populationSize
        self.generations = generations
        self.maxSize = maxSize # This is the max length that an architecture can be
        self.imageColor = imageColor # The number of color channels in the input images
        self.population = [] # The current plan is for this to contain the active population with another being used for all manually trained to avoid training two identical architectures
        self.currentGeneration = []
        self.gaOps = gaOperands(crossOverRate, mutationRate)
    
    """
    Function Name: evolve
    Description: This function is the main loop for the evolution process. At first it generates the intiial population.
    Parameter: 
        None
    Return: 
        None
    """
    def evolve (self):
        self.currentGeneration = ga.generateArchitectures(self.populationSize, self.maxSize)

        #trainSet, testSet, classes = rename.DataLoaders.load_data()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # A test cnn architecture will just be 1 convblock
        # model = LayerBlocks.model(self.population[0], 9, self.imageColor)

        # Just filler comment for where the architecture training / evaluation would be (original code is further down the file)

        # Cross over has a cross over probaility
        # for now this will be hard set and more meant to represent an example
        for candidate in self.currentGeneration:
            candidate.setFitness(random.randint(0, 100)) 
        
        # Now perform the evolutionary loop
        for generation in range (self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            
            for candidate in self.currentGeneration:
                candidate.setFitness(random.randint(0, 100))
                model = LayerBlocks.model(candidate, 9, self.imageColor)
                #summary(model, input_size=(1, 1, 28, 28))
                self.population.append(candidate)
            self.selection()
        os._exit(0)

        # model.to(device)
        # os._exit(0) # This is just to exit the program after the summary is printed. This will be removed once actual training is implemented.
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # criterion = torch.nn.CrossEntropyLoss()
        # out = tqdm(trainSet, desc="Training Progress")
        # totalLoss = 0
        # for epoch in range(2):
        #     model.train()
        #     for image, label in out:
        #         image, label = image.to(device), label.to(device)
        #         optimizer.zero_grad()
        #         output = model(image)
        #         loss = criterion(output, label)
        #         loss.backward()
        #         optimizer.step()
        #         totalLoss += loss.item()
        #         out.set_postfix({"Loss": totalLoss / (len(trainSet) * (epoch + 1))})
        #         out.set_description(f"Epoch {epoch+1}/{2}")

    """
    Function Name: crossover
    Description: This function creates 2 offspring architectures performing uniform crossover on two parent architectures
    Parameter:
        parent1: The first parent architecture
        parent2: The second parent architecture
    Return:
        offspring1: The first offspring architecture
        offspring2: The second offspring architecture
    """
    def UniformCrossover (self, parent1, parent2):
        # Check if crossover is performed based on the crossover rate. If not than just return the parents as offspring
        if random.random() < self.crossOverRate:
            # Currently there is no determined selection criteria performed before this so parent 1 is the better "fitness" architecture. May potenitally allow for parent 1 to be the same as parent 2 but this is currently just testing
            offspring1 = []
            offspring2 = []

            # Loop through each gene and randomly select which parent to take each gene from. Currently 50/50 chance
            for i in range ((self.maxSize + 1)):
                genePassed = random.random()
                if genePassed < 0.5:
                    offspring1.append(parent1.getFullArch()[i])
                    offspring2.append(parent2.getFullArch()[i])
                else:
                    offspring1.append(parent2.getFullArch()[i])
                    offspring2.append(parent1.getFullArch()[i])
            
            # Turn the generated offsprings into architecture objects, then check if they are duplicates
            offspring1 = arch(offspring1)
            offspring2 = arch(offspring2)
            
            return offspring1, offspring2
        else:
            return parent1, parent2

    
    def selection (self):
        """
            This will be used to select candidates that will undergo crossover and eventually replace the filler loop inside the evolve function.
        """

        # This may become a switch for the type of selection strategy being utilised but for now will just call the tournament selection function
        self.tournamentSelection()
    
        return None
    
    def tournamentSelection (self):
        """
            Note: This is currently binary tournament selection.
            Currently tester / filler function for selection. The seperation from the selection funciton was selected to support multiple selection strategies if wanted
        """
        newPopulation = []
        # Loop for the required number of new offspring to be generated. This is currently hard coded to 2 for testing
        for i in range (self.populationSize // 2):
            # Make a deep copy to avoid any changes to the original population. Then select the 2 candidates
            selectionPopulation = copy.deepcopy(self.currentGeneration)

            candidate1 = self.selectCandidate(selectionPopulation)
            selectionPopulation.remove(candidate1)

            candidate2 = self.selectCandidate(selectionPopulation)

            del selectionPopulation

            # perform crossover on the selected candidates and then store them (Currently not implemented, just generating the new offspring)
            offspring1, offspring2 = self.gaOps.crossover(candidate1, candidate2)

            offspring1, offspring2 = self.gaOps.mutation(offspring1), self.gaOps.mutation(offspring2)

            # duplicates should be checked for after mutating

            newPopulation.append(offspring1)
            newPopulation.append(offspring2)
        del self.currentGeneration
        self.currentGeneration = newPopulation

    def selectCandidate (self, selectionPopulation):

        # select 2 candidates from the population to compete to be a parent
        try:
            selectedCandidates = random.sample(selectionPopulation, k=2)
        except ValueError:
            # If the population size is 2, then after the first candidate is selected and removed from the selection pool then
            # an error will be thrown when sampling as there is not enough candidates to select from
            if len(selectionPopulation) == 1:
                return selectionPopulation[0]
            else:
                raise ValueError("Not enough candidates in the selection population to select 2 candidates.")
        candidate1 = selectedCandidates[0]
        candidate2 = selectedCandidates[1]

        # Return the candidate with the better fitness
        if candidate1.getFitness() > candidate2.getFitness():
            return candidate1
        else:
            return candidate2