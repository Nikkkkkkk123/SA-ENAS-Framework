"""
File Name: Evolve.py
Description: This file is the main evolutionary file containing the main loop.
"""
import rename
from GenerateArchitecture import GenerateArchitecture as ga
from ArchitectureCodec import ArchitectureCodec as ac
import LayerBlocks
import torch
from tqdm import tqdm
import os

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
        activeLayers = ac.activeLayers(self.population[0])
        print(f"Initial Population: {self.population}")
        print(f"Active Layers: {activeLayers}")

        trainSet, testSet, classes = rename.DataLoaders.load_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # A test cnn architecture will just be 1 convblock
        model = LayerBlocks.model(activeLayers, len(classes), self.imageColor)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        out = tqdm(trainSet, desc="Training Progress")
        totalLoss = 0
        for epoch in range(2):
            model.train()
            for image, label in out:
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                totalLoss += loss.item()
                out.set_postfix({"Loss": totalLoss / (len(trainSet) * (epoch + 1))})
                out.set_description(f"Epoch {epoch+1}/{2}")
        

