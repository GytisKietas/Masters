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

def save(t_values, states, args, metadata):
    save_dir = args.savedir if args.savedir else "."
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{args.name}.csv")

    meta_str = ""
    for key, value in metadata.items():
        if key != "header":  # Don't duplicate the column names
            meta_str += f"# {key}: {value}\n"

    column_names = ",".join(metadata["header"])
    full_header = f"{meta_str}{column_names}"

    data = np.column_stack((t_values, states))

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=full_header,
        comments=""
    )

    logger.info(f"Saved results and metadata to {filename}")


def save_checkpoint(name, t_values, states, header):
    save_dir = "./CHECKPOINT"
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{name}.csv")

    data = np.column_stack((t_values, states))

    header = ["t"] + header
    header_line = ",".join(header)

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header_line,
        comments=""
    )