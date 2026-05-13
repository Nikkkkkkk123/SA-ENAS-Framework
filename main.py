# terminal input parameters
import argparse
from architecture import Architecture

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA-ENAS Framework")
    
    parser.add_argument("--population_size", type=int, default=10, help="The size of the population for each generation.")
    parser.add_argument("--architecture_length", type=int, default=5, help="The number of layers in each architecture.")

    args = parser.parse_args()

    architecture = Architecture(args.architecture_length)