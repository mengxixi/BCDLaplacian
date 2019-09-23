import os
import matplotlib
matplotlib.use('agg')

import numpy as np
import train
import parse_args

from itertools import product
from base import utils as ut
from base import plot

ROOT = "/home/siyi/tmp"

loss2name = {"ls": "Least Squares", 
             "lg":"Logistic", 
             "sf":"Softmax", 
             "bp":"LP_Quadratic",
             "bp_huber":"LP_Huber",
             "lsl1nn":"Non-negative Least Squares",}

if __name__ == "__main__":
    argsList = parse_args.parse()

    # DEFINTE PATHs
    plots_path = os.path.join(ROOT, "Checkpoints/CoordinateDescent/Figures")
    logs_path = os.path.join(ROOT, "Checkpoints/CoordinateDescent/Logs")
    datasets_path = os.path.join(ROOT, "Datasets/CoordinateDescent")

    # LOOP OVER EXPERIMENTS
    for args in argsList:   
        plotList = []
        historyDict = {}

        ######## TRAIN STAGE #########

        # Loop over datasets
        for dataset_name, loss_name in zip(args.dataset_names, 
                                           args.loss_names):
            figureList = []
            # Loop over loss names
            for block_size in args.blockList:  

                traceList = []

                # Loop over p, s, and u rules
                full_updates = []
                block_updates = args.u_rules.copy()
                for rule in args.u_rules:
                  if "-full" in rule:
                    block_updates.remove(rule)
                    full_updates.append((None, None, rule))

                combinations = list(product(args.p_rules, args.s_rules,
                                       block_updates))
                combinations.extend(full_updates)

                for p, s, u in combinations:

                    # Ignore the following combinations
                    if "-full" not in u and ((p != "VB" and s == "BGSC") or
                        (p != "VB" and s == "OMP") or
                        (p == "VB" and s == "GSQ") or
                        (p != "VB" and "GSQ-" in s) or
                        (p != "VB" and s == "GSC") or
                        (p != "VB" and s == "Perm") or
                        (p == "VB" and s == "BGSL") or
                        (p == "VB" and s == "GSL") or
                        (p == "VB" and s == "cCyclic")or
                        (p == "VB" and s == "Cyclic" and u == "LS") or
                        (p == "Order" and s == "TreePartitions") or
                        (p != "VB" and s == "IHT")or
                        (p != "VB" and s == "GSDHb")):
                        continue

                    if p == "VB":
                      block_size = -1
                    history = train.train(dataset_name=dataset_name,
                                          loss_name=loss_name,
                                          block_size=block_size,
                                          partition_rule=p,
                                          selection_rule=s,
                                          update_rule=u,
                                          n_iters=args.n_iters,
                                          reset=args.reset,
                                          L1=args.L1,
                                          L2=args.L2,
                                          root=ROOT,
                                          logs_path=logs_path,
                                          datasets_path=datasets_path)

                    avg_update_time = history["time"].values[-1]
                    legend = ut.legendFunc(p, s, u, args.p_rules, args.s_rules,
                                           args.u_rules, args.plot_names)

                    legend = "%s" % legend.strip("-None")
                    if args.plot_time:
                      legend += "\n%.2fs/it" % avg_update_time

                    if "converged" in history.columns:
                      ind = np.where(np.isnan(np.array(history["converged"])))[0][-1] + 1
                      converged = {"Y":history["converged"][ind],
                                   "X":ind}
                    else:
                      converged = None

                    loss = np.array(history["loss"])
                    traceList += [{"Y":loss, 
                                   "X":np.array(history["iteration"]),
                                   "legend":legend,
                                   "converged":converged}]

                    historyDict[legend] = history

                if block_size == -1:
                  xlabel = "Iterations"
                else:
                  xlabel = "Iterations with %d-sized blocks" % block_size
                
                ylabel = "$f(x) - f^*$ for %s" % loss2name[loss_name]
                if len(args.dataset_names) > 1:
                  ylabel += " on Dataset %s" % dataset_name.upper()

                figureList += [{"traceList":traceList,
                                "xlabel":xlabel,
                                "ylabel":ylabel,
                                "yscale":"log"}]
            
            plotList += [figureList]

            

        for i, figlist in enumerate(plotList[0]):
          for j, f in enumerate(figlist["traceList"]):
            if i==0 and f["legend"] =="Exact-Cyclic-Order":
              f["legend"] = "Exact $O(n)$"
            elif i==1 and f["legend"] =="Exact-Cyclic-Order":
              f["legend"] = "Exact $O(n^2)$"
              plotList[0][0]["traceList"].append(f)
              figlist["traceList"].pop(j)
            elif i==0 and f["legend"] == "Exact-Tree-VB":
              f["legend"] = "Tree $O(n)$"
            elif i==0 and f["legend"] == "Laplacia":
              f["legend"] = "Laplacian $O(n)$"
            else:
              figlist["traceList"].pop(j)

        plotList[0].pop(1)

        ########## PLOT STAGE #########
        fig = plot.plot(plotList, expName=args.expName, path=plots_path)

        # ut.visplot(fig, win=args.expName)
        matplotlib.pyplot.close()




        ########## SAVE EXP HISTORY ##########
        ut.save_pkl(logs_path + args.expName + ".pkl", historyDict)