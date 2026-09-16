import gc

import torch 
import torch.nn as nn
from Architecture import Architecture as arch
from GA_ops import GA_ops as ga
from Dataset import Dataset as dl
import LayerBlocks
import random
from torchinfo import summary
import os

import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from Encode import Encode as encode
import torch.nn.functional as F
from datetime import datetime
import Surrogate
import numpy as np

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    _crossoverRate: float
    _device: str
    _surrogateEnabled: bool
    _dataset: str

    def __init__ (self, populationSize: int, maxSize: int, inputChannels: int, noGenerations: int, imageSize: int, batchSize: int, epochs: int, 
                  mutationRate: float, crossoverRate: float, surrogateEnabled: bool, dataset: str) -> None:
        self._populationSize = populationSize
        self._maxSize = maxSize
        self._inputChannels = inputChannels
        self._noGenerations = noGenerations
        self._imageSize = imageSize
        self._mutationRate = mutationRate
        self._crossoverRate = crossoverRate
        self._batchSize = batchSize
        self._epochs = epochs
        self._surrogateEnabled = surrogateEnabled

        self.bestModels = []

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.trainDL, self.valDL, self.testDL, self.classNames = dl().getDataset(self._batchSize, self._imageSize, self._inputChannels, dataset)
        self._currentGeneration = []
        self._entirePopulation = []
        self._bestModel = None
        self.surrogate = Surrogate.Surrogate()
        self.log = open("log.txt", "w")

        self.surTrain = []
        self.surLabels = []

    def evolve (self) -> None:
        self.log.write(f"{datetime.now()}: Starting the evolutionary process\n")
        self.log.flush()

        self.generateInitialPopulation()

        self.log.write(f"{datetime.now()}: Training initial population\n")
        self.log.flush()
        self.runCurrentGenModel(self._currentGeneration)

        for generation in range (1, self._noGenerations):
            print(f"Generation {generation}/{self._noGenerations}")
            self.log.write(f"{datetime.now()}: Generation {generation}/{self._noGenerations}\n")
            self.log.flush()

            # Perform crossover and mutation to produce offspring for the next generation
            self.runGA()

            # If the surrogate is enabled then it will be trained and then predict the current generations fitness
            # if it is not enabled then it will just use the current generation
            if self._surrogateEnabled:
                self.surrogate.fit(np.asarray(self.surTrain), np.asarray(self.surLabels))
                beingEval = self._predictFitnessForCurrentGen()
            else:
                beingEval = self._currentGeneration

            # It will manually evaluate the selected candidates. It will then add the evaulated candidates to the surrogate training data if the surrogate is enabled
            self.runCurrentGenModel(beingEval)

        self.log.write(f"{datetime.now()}: Finished evolutionary process\n")
        self.log.flush()

        # now have the best model it needs to be tested
        self.bestModels.sort(key=lambda x: x[1], reverse=True)

        self.log.write(f"{datetime.now()}: Best identified model architecture {self.bestModels[0][0].getEncodedArchitecture()}, Fitness: {self.bestModels[0][1]:.4f}\n")
        self.log.flush()

        self.log.write(f"{datetime.now()}: List of best models\n")
        self.log.flush()
        for model, f1 in self.bestModels:
            self.log.write(f"Architecture: {model.getEncodedArchitecture()}, Fitness: {(f1):.2f}%\n, Number of Parameters: {model.getNoParameters()}\n")
            self.log.flush()

    def final_evaluation (self, encoded_model: list[int]) -> None:
        self.log.write(f"{datetime.now()}: Starting final evaluation of model {encoded_model}\n")
        self.log.flush()

        final_architecture = arch(self._maxSize, self._inputChannels, self._imageSize)
        if not final_architecture.buildEncodedArchitecture(encoded_model):
            self.log.write(f"{datetime.now()}: Invalid encoded model provided for final evaluation: {encoded_model}\n")
            self.log.flush()
            raise ValueError(f"Invalid encoded model provided for final evaluation: {encoded_model}")


        model = LayerBlocks.model(final_architecture, len(self.classNames), self._inputChannels, self._imageSize)
        final_architecture.setNoParameters(sum(p.numel() for p in model.parameters() if p.requires_grad))
        f1ScoreResult: float = 0.0

        model.to(self._device)
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        self.runEpochs(model, optimizer, criterion, final_architecture)
        self.log.write(f"{datetime.now()}: Final evaluation of model {encoded_model} has a fitness of {(final_architecture.getFitness()):.2f}%\n")
        self.log.flush()

        self._testModel(model, final_architecture)

    def  runCurrentGenModel (self, evaluateCandidateList: list) -> None:
        no_candidate = 1
        total_candidates = len(evaluateCandidateList)
        for candidate in evaluateCandidateList:
            # This is a just in case. this should not occur
            if candidate.getTrained():
                self.log.write(f"{datetime.now()}: Candidate {candidate.getEncodedArchitecture()} has already been evaluated with a fitness of {candidate.getFitness()}\n")
                continue
            self.log.write(f"{datetime.now()} ({no_candidate} / {total_candidates}): Evaluating candidate {candidate.getEncodedArchitecture()}\n")
            model = LayerBlocks.model(candidate, len(self.classNames), self._inputChannels, self._imageSize)
            candidate.setNoParameters(sum(p.numel() for p in model.parameters() if p.requires_grad))
            f1ScoreResult: float = 0.0

            model.to(self._device)
            model = nn.DataParallel(model)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            self.runEpochs(model, optimizer, criterion, candidate)
            self._addToBestModels(candidate, candidate.getFitness())

            # add the evaluated candidate to the surrogate model. If the surrogate is not enabled then the function will not do anything
            self._addSurTrain(candidate)
            no_candidate += 1
            del model, optimizer, criterion
            gc.collect()
            torch.cuda.empty_cache() 

    def runEpochs (self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, candidate: arch) -> None:
        for epoch in range(self._epochs):
            loop = tqdm(enumerate(self.trainDL, 0), total=len(self.trainDL), desc=f"Training: Epoch {epoch + 1} / {self._epochs}", colour="blue")
            self.log.write(f"{datetime.now()}: Training: Epoch {epoch + 1} / {self._epochs} || ")
            #print(self.trainDL.dataset.dataset.imgs[0][0])
            self.log.flush()

            self._trainModel(model, optimizer, criterion, loop)
            loop = tqdm(enumerate(self.valDL, 0), total=len(self.valDL), desc=f"Validation: Epoch {epoch + 1} / {self._epochs}", colour="blue")
            self.log.write(f"{datetime.now()}: Validation: Epoch {epoch + 1} / {self._epochs} || ")
            self.log.flush()

            f1ScoreResult = self._validateModel(model, criterion, loop)
            loop.close()
        overall = (f1ScoreResult * 100) # dont want to compute every time but this is mainly for testing at this point

        candidate.calculateFitness(overall)
        candidate.setTrained(True)
        self.log.write(f"{datetime.now()}: Evaluated candidate has a fitness of {candidate.getFitness()}\n")
        self.log.flush()

    def _trainModel (self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loop: tqdm) -> None:
        model.train()
        accuracy: float = 0.0
        f1: float = 0.0
        allPreds, allTargets = [], []
        for i, (images, labels) in loop:
            inputs, labels = images.to(self._device), labels.to(self._device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            allPreds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            allTargets.append(labels.cpu().numpy())
            accuracy = accuracy_score(np.concatenate(allTargets), np.concatenate(allPreds))
            f1 = f1_score(np.concatenate(allTargets), np.concatenate(allPreds), average='weighted')

            loop.set_postfix(Accuracy=f"{accuracy:.4f}", F1_Score=f"{f1:.4f}")
        

        self.log.write(f"Accuracy: {(accuracy * 100):.2f}% || F1_Score: {(f1 * 100):.2f}%\n")
        self.log.flush()
        del allPreds, allTargets, accuracy, f1

    def _validateModel (self, model: torch.nn.Module, criterion: torch.nn.Module, loop: tqdm) -> float:
        model.eval()
        accuracyResult: float = 0.0
        f1ScoreResult: float = 0.0
        allPreds, allTargets = [], []

        with torch.no_grad():
            for i, (images, labels) in loop:
                inputs, labels = images.to(self._device), labels.to(self._device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                allPreds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                allTargets.append(labels.cpu().numpy())
                accuracyResult = accuracy_score(np.concatenate(allTargets), np.concatenate(allPreds))
                f1ScoreResult = f1_score(np.concatenate(allTargets), np.concatenate(allPreds), average='weighted')
                loop.set_postfix(Accuracy=f"{accuracyResult:.4f}", F1_Score=f"{f1ScoreResult:.4f}")
        self.log.write(f"Accuracy: {(accuracyResult * 100):.2f}% || F1_Score: {(f1ScoreResult * 100):.2f}%\n")
        self.log.flush()
        del allPreds, allTargets
        return f1ScoreResult

    def _testModel (self, model: torch.nn.Module, candidate: arch) -> float:
        model.eval()
        self.log.write(f"{datetime.now()}: Begging final test of model\n")
        self.log.flush()
        loop = tqdm(enumerate(self.testDL, 0), total=len(self.testDL), desc=f"Testing Architecture: ", colour="blue")
        accuracyResult: float = 0.0
        f1ScoreResult: float = 0.0
        allPreds, allTargets = [], []

        with torch.no_grad():
            for i, (images, labels) in loop:
                inputs, labels = images.to(self._device), labels.to(self._device)

                outputs = model(inputs)

                allPreds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                allTargets.append(labels.cpu().numpy())
                accuracyResult = accuracy_score(np.concatenate(allTargets), np.concatenate(allPreds))
                f1ScoreResult = f1_score(np.concatenate(allTargets), np.concatenate(allPreds), average='weighted')
                loop.set_postfix(Accuracy=f"{accuracyResult:.4f}", F1_Score=f"{f1ScoreResult:.4f}")
        self.log.write(f"Final test results of model {candidate.getEncodedArchitecture()}\n")
        self.log.write(f"No Parameters: {candidate.getNoParameters()}, F1 score: {f1ScoreResult:.4f}\n")
        self.log.write(f"Final Test Accuracy: {(accuracyResult * 100):.2f}% || Final Test F1_Score: {(f1ScoreResult * 100):.2f}%\n")
        self.log.write(f"Confusion Matrix:\n{confusion_matrix(np.concatenate(allTargets), np.concatenate(allPreds))}\n")
        self.log.write(f"Classification Report:\n{classification_report(np.concatenate(allTargets), np.concatenate(allPreds), target_names=self.classNames)}\n")
        self.log.flush()
        self.log.write(f"Testing Finished\n")
        del allPreds, allTargets
        return f1ScoreResult


    def runGA (self) -> None:
        # We want keep x number of the top performing candidates and generate the rest through crossover and mutation

        tempGeneration: list[arch] = []

        # For now going to hard code that the top 2 candidates are kept for the next generation.
        self.bestModels.sort(key=lambda x: x[1], reverse=True)
        tempGeneration.append(self.bestModels[0][0])
        tempGeneration.append(self.bestModels[1][0])

        for i in range ((self._populationSize - 2) // 2):
            offspring1: arch = None
            offspring2: arch = None

            while offspring1 is None or offspring2 is None:

                offspring1, offspring2 = ga.performGA(self._currentGeneration, self._maxSize, self._inputChannels, self._imageSize, self._mutationRate, self._crossoverRate)

                # If they are duplicates it currently remakes from scratch with different parents, this should be changed
                if self.checkDuplicate(offspring1.getActiveEncoding()) and self.checkDuplicate(offspring2.getActiveEncoding()):
                    tempGeneration.append(offspring1)
                    tempGeneration.append(offspring2)
                    self.addToEntirePopulation(offspring1)
                    self.addToEntirePopulation(offspring2)
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
                    self.addToCurrentGen(newArch)
                    self._entirePopulation.append(newArch.getActiveEncoding())
                else:
                    newArch = None

    def addToCurrentGen (self, newArch: arch) -> None:
        self._currentGeneration.append(newArch)

    def addToEntirePopulation (self, newArch: arch) -> None:
        self._entirePopulation.append(newArch.getActiveEncoding())

    def updateCurrGen (self, newGen: list[arch]) -> None:
        del self._currentGeneration
        self._currentGeneration = newGen

    def checkDuplicate (self, newArch: list) -> bool:
        if newArch in self._entirePopulation:
            # print(f"Duplicate architecture found: {newArch}")
            return False
        return True

    def _addToBestModels (self, architecture: arch, f1: float) -> None:
        if len(self.bestModels) == 0:
            self.bestModels.append((architecture, f1))
            return
        if len(self.bestModels) >= 5:
            if f1 > self.bestModels[-1][1]:
                self.bestModels.remove(self.bestModels[-1])
            else:
                return
        self.bestModels.append((architecture, f1))
        self.bestModels.sort(key=lambda x: x[1], reverse=True)

    def _addSurTrain (self, candidate: arch) -> None:
        if self._surrogateEnabled:
            self.surTrain.append(candidate.getEncodedArchitecture())
            self.surLabels.append(candidate.getFitness())

    def _predictFitnessForCurrentGen (self) -> list[arch]:
        for candidate in self._currentGeneration:
            # We dont want to predict the fitness of a candidate that has already been manually evaluated
            if not candidate.getTrained():
                candidate.predictFitness(self.surrogate)
                self._entirePopulation.remove(candidate.getActiveEncoding())

        # Return the top 10% candidates to be manually evaluated. If it is 0 then return only the top candidate
        return self._selectedEvalCandidates()

    def _selectedEvalCandidates (self) -> list[arch]:
        self._currentGeneration.sort(key=lambda x: x.getFitness(), reverse=True)
        if len(self._currentGeneration) // 10 == 0:
            return [self._currentGeneration[0]]

        bestUntrained = [candidate for candidate in self._currentGeneration if not candidate.getTrained()]
        bestUntrained = bestUntrained[:5]
        for candidate in bestUntrained:
            self.addToEntirePopulation(candidate)
        return bestUntrained # this will return the top 5 candidates that have not been manually evaluated
