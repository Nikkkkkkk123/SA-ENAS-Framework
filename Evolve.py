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
from sklearn.metrics import classification_report, f1_score

class Evolve:
    _populationSize: int
    _maxSize: int
    _inputChannels: int
    _currentGeneration: list[arch]
    _noGenerations: int
    _imageSize: int
    _batchSize: int
    _epochs: int
    _bestModel: arch
    _mutationRate: float
    _device: str

    def __init__ (self, populationSize: int, maxSize: int, inputChannels: int, noGenerations: int, imageSize: int, batchSize: int, epochs: int, mutationRate: float) -> None:
        self._populationSize = populationSize
        self._maxSize = maxSize
        self._inputChannels = inputChannels
        self._noGenerations = noGenerations
        self._imageSize = imageSize
        self._mutationRate = mutationRate
        self._batchSize = batchSize
        self._epochs = epochs

        self.bestModels = []

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.trainDL, self.valDL, self.testDL, self.classNames = dl().getDataset(self._batchSize, self._imageSize, self._inputChannels)
        self._currentGeneration = []
        self._entirePopulation = []
        self._bestModel = None

    def evolve (self) -> None:

        self.generateInitialPopulation()
        self.runCurrentGenModel()

        for generation in range (1, self._noGenerations + 1):
            print(f"Generation {generation}/{self._noGenerations}")
            self.runGA()

            self.runCurrentGenModel()

        # now have the best model it needs to be tested
        self.bestModels.sort(key=lambda x: x[1], reverse=True)
        self._bestModel = self.bestModels[0][0]
        if self._bestModel is not None:
            model = LayerBlocks.model(self._bestModel, len(self.classNames), self._inputChannels)      
            allPreds, allTargets = [], []
            loop = tqdm(enumerate(self.testDL, 0), total=len(self.testDL), desc=f"Testing: Best Model", colour="blue")
            allPreds, allTargets = self._validateModel(model, torch.nn.CrossEntropyLoss(), self._device, loop)
            print(classification_report(allTargets, allPreds, target_names=self.classNames, digits=4, zero_division=0))

    def runCurrentGenModel (self) -> None:
        for candidate in self._currentGeneration:
            model = LayerBlocks.model(candidate, len(self.classNames), self._inputChannels)
            candidate.setNoParameters(sum(p.numel() for p in model.parameters() if p.requires_grad))
            model.to(self._device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(self._epochs):
                loop = tqdm(enumerate(self.trainDL, 0), total=len(self.trainDL), desc=f"Training: Epoch {epoch + 1} / {self._epochs}", colour="blue")
                self._trainModel(model, optimizer, criterion, loop)

                loop = tqdm(enumerate(self.valDL, 0), total=len(self.valDL), desc=f"Validation: Epoch {epoch + 1} / {self._epochs}", colour="blue")

                allPreds, allTargets = self._validateModel(model, criterion, loop)

            overall = (f1_score(allPreds, allTargets, average="weighted") * 100)
            candidate.calculateFitness(overall)
            self.bestModels.append((candidate, overall))

    def _trainModel (self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loop: tqdm) -> tuple[float, float]:
        model.train()
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.classNames)).to(self._device)
        f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(self.classNames), average="weighted").to(self._device)
        for i, (images, labels) in loop:
            inputs, labels = images.to(self._device), labels.to(self._device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            accuracy.update(outputs, labels)
            f1_score.update(outputs, labels)

            loop.set_postfix(Accuracy=f"{accuracy.compute().item():.4f}", F1_Score=f"{f1_score.compute().item():.4f}")

    def _validateModel (self, model: torch.nn.Module, criterion: torch.nn.Module, loop: tqdm) -> tuple[list[int], list[int]]:
        model.eval()
        allPreds = []
        allTargets = []
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.classNames)).to(self._device)
        f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(self.classNames), average="weighted").to(self._device)
        with torch.no_grad():
            for i, (images, labels) in loop:
                inputs, labels = images.to(self._device), labels.to(self._device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)


                accuracy.update(outputs, labels)
                f1_score.update(outputs, labels)

                loop.set_postfix(Accuracy=f"{accuracy.compute().item():.4f}", F1_Score=f"{f1_score.compute().item():.4f}")

                for k in range (labels.size(0)):
                    allPreds.append(outputs[k].argmax().item())
                    allTargets.append(labels[k].item())
                
        return allPreds, allTargets


    def runGA (self) -> None:
        tempGeneration: list[arch] = []
        counter = 0
        for i in range (self._populationSize // 2):
            offspring1: arch = None
            offspring2: arch = None

            while offspring1 is None or offspring2 is None:
                counter += 1
                parent1, parent2 = ga.selectParents(self._currentGeneration)

                offspring1, offspring2 = ga.crossover(parent1, parent2, self._maxSize, self._inputChannels, self._imageSize, self._mutationRate)

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