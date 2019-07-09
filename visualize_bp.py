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

    subexps = history_list.keys()
    for subexp in subexps:
        history = history_list[subexp]
        dataset = datasets.load(args.dataset_names[0], path=datasets_path)
        _, _, dargs = dataset["A"], dataset["b"], dataset["args"]

        y = dargs["ytrue"]
        ybar = y[dargs["unlabeled"]]
        # print(history["loss"].values[-1])
        print(history["x"].values[-1]) 
        # print(ybar)   
