import argparse
import logging
from simulation import harmonic_rk4
from simulation import phi_rk4
from simulation import slowroll
from simulation import phi_homogeneous
from utils import io

simulations = {
    "harmonic" : harmonic_rk4,
    "phi" : phi_rk4,
    "slowroll" : slowroll,
    "homogeneous" : phi_homogeneous
}


def parse_args():
    logging.info("Parsing arguments...")
    parser = argparse.ArgumentParser(description="Run numerical experiments")
    parser.add_argument("--sim", type=str, default = "coupled")
    parser.add_argument("--name", type=str, default = "rk4")
    parser.add_argument("--savedir", type=str, default = "/home/gytis/Masters/thesis/baryogenesis/code/SAVED_HERE/")
    parser.add_argument("--ic", type=str, default = "./ic/test.json")

    args = parser.parse_args()

    logging.debug(f"sim: {args.sim}")
    logging.debug(f"name: {args.name}")
    logging.debug(f"save dir: {args.savedir}")
    logging.debug(f"ic: {args.ic}")

    return args


def run_experiment(args, ic):
    logging.info(f"Running experiment: {args.sim} as \"{args.name}\" ")
    return simulations[args.sim].run(args, ic)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    args = parse_args()
    ic = io.load(args)
    logging.info(f"Initial conditions {ic}")
    t_values, states, metadata = run_experiment(args, ic)
    io.save(t_values, states, args, metadata)


if __name__ == "__main__":
    main()