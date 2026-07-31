import torchmetrics
import torch
from Architecture import Architecture as arch
from GA_ops import GA_ops as ga
from Dataset import Dataset as dl
import LayerBlocks
import random
from torchinfo import summary
import os
import copy
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

class Evolve:
    _populationSize: int
    _maxSize: int
    _inputChannels: int
    _currentGeneration: list[arch]
    _noGenerations: int
    _imageSize: int
    _batchSize: int

    def __init__ (self, populationSize: int, maxSize: int, inputChannels: int, noGenerations: int, imageSize: int, batchSize: int) -> None:
        self._populationSize = populationSize
        self._maxSize = maxSize
        self._inputChannels = inputChannels
        self._noGenerations = noGenerations
        self._imageSize = imageSize
        self._batchSize = batchSize
        self.trainDL, self.valDL, self.testDL, self.classNames = dl().getDataset(self._batchSize, self._imageSize, self._inputChannels)
        self._currentGeneration = []
        self._entirePopulation = []

    def evolve (self) -> None:

        self.generateInitialPopulation()
        self.runCurrentGenModel()

        for generation in range (1, self._noGenerations + 1):
            print(f"Generation {generation}/{self._noGenerations}")
            self.runGA()

            self.runCurrentGenModel()

    def runCurrentGenModel (self) -> None:
        for candidate in self._currentGeneration:
            model = LayerBlocks.model(candidate, 9, self._inputChannels)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(5):
                accuracy = 0
                f1_score = 0
                counter = 0
                loop = tqdm(enumerate(self.trainDL, 0), total=len(self.trainDL), desc=f"Epoch {epoch + 1} / {5}, Accuracy: {accuracy:.4f}", ncols=100, colour="blue")
                for i, data in loop:
                    counter += 1
                    if counter > 10:
                        break
                    inputs, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()

                    optimizer.step()

                    newaccuracy = torchmetrics.functional.accuracy(outputs, labels, task="multiclass", num_classes=len(self.classNames))
                    newf1_score = torchmetrics.functional.f1_score(outputs, labels, task="multiclass", num_classes=len(self.classNames), average="macro")

                    if newaccuracy >= accuracy and newf1_score >= f1_score:
                        accuracy = newaccuracy
                        f1_score = newf1_score
                        loop.colour = "blue"
                        loop.set_description(f"Epoch {epoch + 1} / {5}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")
                    else:
                        loop.colour = "red"
                        accuracy = newaccuracy
                        f1_score = newf1_score
                        loop.set_description(f"Epoch {epoch + 1} / {5}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")
                loop = tqdm(enumerate(self.valDL, 0), total=len(self.valDL), desc=f"Epoch {epoch + 1} / {5}, Accuracy: {accuracy:.4f}", ncols=100, colour="blue")
                counter = 0

                allOred = []
                alltargets = []
                for i, data in loop:
                    inputs, labels = data[0].to(device), data[1].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)


                    accuracy = torchmetrics.functional.accuracy(outputs, labels, task="multiclass", num_classes=len(self.classNames))
                    f1_score = torchmetrics.functional.f1_score(outputs, labels, task="multiclass", num_classes=len(self.classNames), average="macro")

                    loop.set_description(f"Epoch {epoch + 1} / {5}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")

                    for i in range (labels.size(0)):
                        allOred.append(torch.argmax(outputs[i]).item())
                        alltargets.append(labels[i].item())
                print(classification_report(alltargets, allOred, target_names=self.classNames))
                os._exit(0)

    def runGA (self) -> None:
        tempGeneration: list[arch] = []
        counter = 0
        for i in range (self._populationSize // 2):
            offspring1: arch = None
            offspring2: arch = None

            while offspring1 is None or offspring2 is None:
                counter += 1
                parent1, parent2 = ga.selectParents(self._currentGeneration)

                offspring1, offspring2 = ga.crossover(parent1, parent2, self._maxSize, self._inputChannels, self._imageSize)

                # If they are duplicates it currently remakes from scratch with different parents, this should be changed
                if self.checkDuplicate(offspring1.getActiveEncoding()) and self.checkDuplicate(offspring2.getActiveEncoding()):
                    tempGeneration.append(offspring1)
                    tempGeneration.append(offspring2)
                else:
                    offspring1 = None
                    offspring2 = None

        # Want to remove the current generation and replace it with the newly generated one
        self.updateCurrGen(tempGeneration)

    def generateInitialPopulation (self) -> None:
        for i in range(self._populationSize):
                    newArch = None
                    while newArch is None:
                        newArch = arch(self._maxSize, self._inputChannels, self._imageSize)
                        newArch.generateArchitecture()

                        if self.checkDuplicate(newArch.getActiveEncoding()):
                            self._currentGeneration.append(newArch)
                            self._entirePopulation.append(newArch.getActiveEncoding())
                        else:
                            newArch = None

    def addToEntirePopulation (self, newArch: arch) -> None:
        self._entirePopulation.append(newArch.getActiveEncoding())

    def updateCurrGen (self, newGen: list[arch]) -> None:
        del self._currentGeneration
        self._currentGeneration = newGen

    def checkDuplicate (self, newArch: list) -> bool:
        if newArch in self._entirePopulation:
            return False
        return True