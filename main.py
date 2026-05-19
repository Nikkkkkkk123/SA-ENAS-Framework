# terminal input parameters
import argparse
from Evolve import Evolve as ev

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA-ENAS Framework")
    
    parser.add_argument("--population_size", type=int, default=10, help="The size of the population for each generation.")
    parser.add_argument("--generations", type=int, default=10, help="The number of generations to evolve.")
    parser.add_argument("--architecture_length", type=int, default=10, help="The number of layers in each architecture.")

    args = parser.parse_args()

    evolution = ev(args.population_size, args.generations, args.architecture_length)
    evolution.evolve()