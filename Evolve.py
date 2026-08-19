import torchmetrics
import torch
from torchvision import transforms
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
from Encode import Encode as encode
import torch.nn.functional as F
from datetime import datetime

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

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainDL, self.valDL, self.testDL, self.classNames = dl().getDataset(self._batchSize, self._imageSize, self._inputChannels)
        self._currentGeneration = []
        self._entirePopulation = []
        self._bestModel = None

        self.log = open("log.txt", "w")
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.classNames)).to(self._device)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(self.classNames), average="weighted").to(self._device)

    def evolve (self) -> None:
        self.log.write(f"{datetime.now()}: Starting the evolutionary process\n")
        self.log.flush()

        self.generateInitialPopulation()

        self.log.write(f"{datetime.now()}: Training initial population\n")
        self.log.flush()
        self.runCurrentGenModel()

        for generation in range (1, self._noGenerations + 1):
            print(f"Generation {generation}/{self._noGenerations}")
            self.log.write(f"{datetime.now()}: Generation {generation}/{self._noGenerations}\n")
            self.log.flush()
            self.runGA()

            self.runCurrentGenModel()

        self.log.write(f"{datetime.now()}: Finished evolutionary process\n")
        self.log.flush()

        # now have the best model it needs to be tested
        self.bestModels.sort(key=lambda x: x[1], reverse=True)
        self._bestModel = self.bestModels[0][0].getModel()

        self.log.write(f"{datetime.now()}: Best identified model architecture {self.bestModels[0][0].getEncodedArchitecture()}, Fitness: {self.bestModels[0][1]:.4f}\n")
        self.log.flush()
        # allPreds, allTargets = [], []

        if self._bestModel is not None:
            model = self._bestModel  
            # model.to(self._device)
            # loop = tqdm(enumerate(self.testDL, 0), total=len(self.testDL), desc=f"Testing: Best Model", colour="blue")
            # allPreds, allTargets = self._validateModel(model, torch.nn.CrossEntropyLoss(), loop)

            # self.log.write(f"{datetime.now()}: Best identified architecture F1 score: {(self.f1_score.compute().item() * 100):.2f}%\n")
            # self.log.flush()

            # report = classification_report(allTargets, allPreds, target_names=self.classNames, digits=4, zero_division=0)

            # self.log.write(f"{datetime.now()}: Classification Report:\n{report}\n")
            # store saved version of fully trained model
            torch.save(model.state_dict(), "best_model.pth")

        self.log.write(f"{datetime.now()}: List of best models\n")
        self.log.flush()
        for model, f1 in self.bestModels:
            self.log.write(f"Architecture: {model.getEncodedArchitecture()}, Fitness: {(f1):.2f}%\n")
            self.log.flush()

    def runCurrentGenModel (self) -> None:
        for candidate in self._currentGeneration:
            self.log.write(f"{datetime.now()}: Evaluating candidate {candidate.getEncodedArchitecture()}\n")
            model = LayerBlocks.model(candidate, len(self.classNames), self._inputChannels)
            candidate.setNoParameters(sum(p.numel() for p in model.parameters() if p.requires_grad))
            model.to(self._device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(self._epochs):
                self.accuracy.reset()
                self.f1_score.reset()
                loop = tqdm(enumerate(self.trainDL, 0), total=len(self.trainDL), desc=f"Training: Epoch {epoch + 1} / {self._epochs}", colour="blue")
                self.log.write(f"{datetime.now()}: Training: Epoch {epoch + 1} / {self._epochs} || ")
                self.log.flush()

                self._trainModel(model, optimizer, criterion, loop)
                self.accuracy.reset()
                self.f1_score.reset()
                loop = tqdm(enumerate(self.valDL, 0), total=len(self.valDL), desc=f"Validation: Epoch {epoch + 1} / {self._epochs}", colour="blue")
                self.log.write(f"{datetime.now()}: Validation: Epoch {epoch + 1} / {self._epochs} || ")
                self.log.flush()

                all_preds, all_targets = self._validateModel(model, criterion, loop)
            overall = (self.f1_score.compute().item() * 100) # dont want to compute every time but this is mainly for testing at this point

            candidate.calculateFitness(overall)
            candidate.setModel(model)
            self.log.write(f"{datetime.now()}: Evaluated candidate has a fitness of {candidate.getFitness()}\n")
            self._addToBestModels(candidate, candidate.getFitness())

    def _trainModel (self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loop: tqdm) -> None:
        model.train()
        for i, (images, labels) in loop:
            inputsOG, labels = images.to(self._device), labels.to(self._device)
            inputsOG.requires_grad = True
            inputs = F.interpolate(inputsOG, size=(self._imageSize, self._imageSize), mode='bilinear', align_corners=False)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            self.accuracy.update(outputs, labels)
            self.f1_score.update(outputs, labels)

            loop.set_postfix(Accuracy=f"{self.accuracy.compute().item():.4f}", F1_Score=f"{self.f1_score.compute().item():.4f}")
        self.log.write(f"Accuracy: {(self.accuracy.compute().item() * 100):.2f}% || F1_Score: {(self.f1_score.compute().item() * 100):.2f}%\n")

    def _validateModel (self, model: torch.nn.Module, criterion: torch.nn.Module, loop: tqdm) -> tuple[list[int], list[int]]:
        model.eval()
        accuracyResult: float = 0.0
        f1ScoreResult: float = 0.0
        allPreds, allTargets = [], []

        for i, (images, labels) in loop:
            inputsOG, labels = images.to(self._device), labels.to(self._device)
            inputs = F.interpolate(inputsOG, size=(self._imageSize, self._imageSize), mode='bilinear', align_corners=False)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            self.accuracy.update(outputs, labels)
            self.f1_score.update(outputs, labels)

            accuracyResult = self.accuracy.compute().item()
            f1ScoreResult = self.f1_score.compute().item()
            loop.set_postfix(Accuracy=f"{accuracyResult:.4f}", F1_Score=f"{f1ScoreResult:.4f}")

            for k in range (labels.size(0)):
                allPreds.append(outputs[k].argmax().item())
                allTargets.append(labels[k].item())

        self.log.write(f"Accuracy: {(accuracyResult * 100):.2f}% || F1_Score: {(f1ScoreResult * 100):.2f}%\n")

        return allPreds, allTargets


    def runGA (self) -> None:
        tempGeneration: list[arch] = []
        for i in range (self._populationSize // 2):
            offspring1: arch = None
            offspring2: arch = None

            while offspring1 is None or offspring2 is None:

                offspring1, offspring2 = ga.performGA(self._currentGeneration, self._maxSize, self._inputChannels, self._imageSize, self._mutationRate)

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

    def _addToBestModels (self, architecture: arch, f1: float) -> None:
        if len(self.bestModels) >= 5:
            self.bestModels.remove(self.bestModels[-1])
        self.bestModels.append((architecture, f1))
        self.bestModels.sort(key=lambda x: x[1], reverse=True)
