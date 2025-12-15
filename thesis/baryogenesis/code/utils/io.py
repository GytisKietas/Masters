import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

def load(args):
    if args.ic:
        logger.info(f"Loading IC from {args.ic}")
        with open(args.ic, "r") as f:
            return json.load(f)
    return None


def save(t_values, states, header, args):

    save_dir = args.savedir if args.savedir else "."
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{args.name}.csv")

    # stack time + state columns
    data = np.column_stack((t_values, states))

    # build header
    header = ["t"] + header
    header_line = ",".join(header)

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header_line,
        comments=""
    )

    logger.info(f"Saved results to {filename}")

