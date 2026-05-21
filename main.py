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
    
    parser.add_argument("--population_size", type=int, default=10, help="The size of the population for each generation.")
    parser.add_argument("--generations", type=int, default=10, help="The number of generations to evolve.")
    parser.add_argument("--architecture_length", type=int, default=10, help="The number of layers in each architecture.")
    parser.add_argument("--image_color", type=int, default=1, help="The number of color channels in the input images.")

    args = parser.parse_args()

    evolution = ev(args.population_size, args.generations, args.architecture_length, args.image_color)
    evolution.evolve()