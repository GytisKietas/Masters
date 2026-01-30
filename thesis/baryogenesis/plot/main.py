import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Plot data from a CSV file.")
    
    parser.add_argument("--csv", type=str, required=True,)
    parser.add_argument("--x", type=str, required=True,)
    parser.add_argument("--y", type=str, nargs="+", required=True,)
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--out", type=str, default="plot.png")

    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    args = parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)

    # Check x column
    if args.x not in data.dtype.names:
        raise ValueError(f"Column '{args.x}' not found in CSV.")

    x = data[args.x]

    plt.figure(figsize=(8, 5))

    for col in args.y:
        if col not in data.dtype.names:
            raise ValueError(f"Column '{col}' not found in CSV.")
        plt.plot(x, data[col], label=col)

    # Axis scale
    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")

    plt.xlabel(args.x)
    plt.ylabel(", ".join(args.y))
    plt.legend()
    plt.grid(True, alpha=0.3)


    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    logging.info(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
