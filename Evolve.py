"""
File Name: Evolve.py
Description: This file is the main evolutionary file containing the main loop.
"""
import rename
from GenerateArchitecture import GenerateArchitecture as ga
from ArchitectureCodec import ArchitectureCodec as ac
from Architecture import Architecture as arch
import LayerBlocks
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
    Return: 
        None
    """
    def __init__(self, populationSize, generations, maxSize, imageColor):
        self.populationSize = populationSize
        self.generations = generations
        self.maxSize = maxSize # This is the max length that an architecture can be
        self.imageColor = imageColor # The number of color channels in the input images
        self.population = [] # The current plan is for this to contain the active population with another being used for all manually trained to avoid training two identical architectures
        # currently no 'used' population implemented
    
    """
    Function Name: evolve
    Description: This function is the main loop for the evolution process. At first it generates the intiial population.
    Parameter: 
        None
    Return: 
        None
    """
    def evolve (self):
        self.population = ga.generateArchitectures(self.populationSize, self.maxSize)

        #trainSet, testSet, classes = rename.DataLoaders.load_data()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # A test cnn architecture will just be 1 convblock
        model = LayerBlocks.model(self.population[0], 9, self.imageColor)

        # Just filler comment for where the architecture training / evaluation would be (original code is further down the file)

        # Cross over has a cross over probaility
        # for now this will be hard set and more meant to represent an example
        for candidate in self.population:
            candidate.setFitness(random.randint(0, 100)) # This is just a placeholder for the actual fitness evaluation which will be done by training the architecture and evaluating its performance on the test set. This is just to check that the fitness is being set correctly and that the architectures are being evolved correctly. This will be removed once actual training is implemented.
        
        crossOverRate = 0.5
        for i in range (self.populationSize):
            # Currently randomly selects parent based on their fitness score as a weight. 
            # My current idea is to implement a roulete selection process so that worse architectures can still be selected as mutations may make them extreamly valuable
            # Potentually a tournement selection strategy may be put into place
            parent1 = random.choices(self.population, weights=[candidate.performance for candidate in self.population], k=1)[0]
            parent2 = random.choices(self.population, weights=[candidate.performance for candidate in self.population], k=1)[0]

            doesCrossOver = random.random() < crossOverRate
            if doesCrossOver:
                p1Arch = parent1.getFullArch()
                p2Arch = parent2.getFullArch()
                crossOverPoint = random.randint(1, 9)
                childArch = p1Arch[:crossOverPoint] + p2Arch[crossOverPoint:]
                print("Parent 1: ", parent1.getFullArch())
                print("\nParent 2: ", parent2.getFullArch())
                print("\nChild: ", childArch)
                childArch = arch(childArch)
                print("\nEncoded Child: ", childArch.getActiveArch())
                os._exit(0)

        # model.to(device)
        # summary(model, input_size=(1, 1, 28, 28)) # This is currently hard coded to the test image size. This is just to check that the model is being initialised correctly and that the active layers are being used correctly. This will be removed once actual training is implemented.
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
        

