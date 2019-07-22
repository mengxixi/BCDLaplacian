import os
import matplotlib
matplotlib.use('agg')

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import parse_args
import pandas as pd

from base import utils as ut
from base import plot
from datasets import datasets


ROOT = "/home/siyi/tmp"
if __name__ == "__main__":
    args = parse_args.parse()[0]
    logs_path = os.path.join(ROOT, "Checkpoints/CoordinateDescent/Logs")
    datasets_path = os.path.join(ROOT, "Datasets/CoordinateDescent")

    log_file = logs_path + args.expName + ".pkl"
    history_list = ut.load_pkl(log_file)

    subexps = list(history_list.values())
    
    losses = []
    for i in range(len(subexps)):
        name = list(history_list.keys())[i]
        if "Greedy Tree" in name:
            history = subexps[i]
            dataset = datasets.load(args.dataset_names[0], path=datasets_path)
            _, _, dargs = dataset["A"], dataset["b"], dataset["args"]

            unlabled_indices = dargs["unlabeled"]

            ybar = history["x"].values[-1]
            # if "ytrue" in dargs.keys() and dargs["ytrue"] is not None:
            #     ytrue = dargs["ytrue"][unlabled_indices].squeeze(1)
            #     print(np.linalg.norm(ybar-ytrue))
            # else:
            #     print(np.min(ybar))
            #     print(np.max(ybar))
                # print(len(ybar), len(ytrue))
            losses.append(history["loss"])
            # print(history["loss"].values[-1])
            # print(np.sum(np.abs(ytrue -ybar)))

    print(losses[0]-losses[1])
