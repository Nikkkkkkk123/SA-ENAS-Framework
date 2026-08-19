# terminal input parameters
import argparse
from Evolve import Evolve as ev
import zlib
import os
import numpy as np
from PIL import Image
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SA-ENAS Framework")
    
    parser.add_argument("--population_size", type=int, default=2, help="The size of the population for each generation.")
    parser.add_argument("--generations", type=int, default=3, help="The number of generations to evolve.")
    parser.add_argument("--architecture_length", type=int, default=10, help="The number of layers in each architecture.")
    parser.add_argument("--input_channels", type=int, default=1, help="The number of input channels in the input images.")
    parser.add_argument("--crossover_rate", type=float, default=0.6, help="The probability of crossover between two parent architectures.")
    parser.add_argument("--mutation_rate", type=float, default=0.4, help="The probability of mutation for each architecture.")
    parser.add_argument("--min_layers", type=int, default=3, help="The minimum number of layers in each architecture.")
    parser.add_argument("--image_size", type=int, default=32, help="The size of the input images (assumed to be square).")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs for training each model.")

    args = parser.parse_args()

    evolution = ev(args.population_size, args.architecture_length, args.input_channels, args.generations, args.image_size, args.batch_size, args.epochs, args.mutation_rate)
    evolution.evolve()