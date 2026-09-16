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
    
    parser.add_argument("--population_size", type=int, default=20, help="The size of the population for each generation.")
    parser.add_argument("--generations", type=int, default=10, help="The number of generations to evolve.")
    parser.add_argument("--architecture_length", type=int, default=10, help="The number of layers in each architecture.")
    parser.add_argument("--input_channels", type=int, default=1, help="The number of input channels in the input images.")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="The probability of crossover between two parent architectures.")
    parser.add_argument("--mutation_rate", type=float, default=0.4, help="The probability of mutation for each architecture.")
    parser.add_argument("--min_layers", type=int, default=3, help="The minimum number of layers in each architecture.")
    parser.add_argument("--image_size", type=int, default=2, help="The size of the input images (assumed to be square).")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs for training each model.")
    parser.add_argument("--surrogate_enabled", action=argparse.BooleanOptionalAction, type=bool, default=True, help="Whether to enable the surrogate model for fitness prediction.")
    parser.add_argument("--dataset", type=str, default="D:", help="Path to the dataset.")
    parser.add_argument("--final_eval", type=bool, default=True, help="Whether you are performing final evaluation on an input model")
    parser.add_argument("--model", nargs='+', type=lambda x: [int(i) for i in x.split(',')], default="2, 0, 0, 1, 2, 1, 1, 0, 4, 3, 1, 1, 0, 5, 2, 3, 1, 1, 0, 2, 2, 4, 1, 1, 1, 2, 5, 5, 4, 1, 3, 5, 2, 0, 1, 1, 7, 4, 2, 1, 1, 8, 3, 2, 1, 7, 8", help="Encoded model string for final evaluation (requered if --final_eval is True)")

    args = parser.parse_args()

    if args.final_eval and args.model is None:
        parser.error("--model is required when --final_eval is True")

    evolution = ev(args.population_size, args.architecture_length, args.input_channels, args.generations, args.image_size, args.batch_size, args.epochs, 
                       args.mutation_rate, args.crossover_rate, args.surrogate_enabled, args.dataset)

    if args.final_eval:
        evolution.final_evaluation(args.model)
    else:
        evolution.evolve()
    