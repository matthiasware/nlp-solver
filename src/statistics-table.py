import utils
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import random

f_ipopt = "ipopt.csv"
f_master = "master.csv"
csvdict_ipopt = utils.getResultsAsDict(f_ipopt)
csvdict_master = utils.getResultsAsDict(f_master)

stat_ipopt = {}
for k, v in csvdict_ipopt.items():
    stat_ipopt[v["msg"]] = stat_ipopt.get(v["msg"], 0) + 1

stat_master = {}
for k, v in csvdict_master.items():
    stat_master[v["msg"]] = stat_master.get(v["msg"], 0) + 1

solved_both = []
solved_none = []
solved_ipopt = []
solved_master = []

for name, item_ipopt in csvdict_ipopt.items():
    item_master = csvdict_master[name]
    if item_ipopt["success"] == "True" and item_master["success"] == "True":
        solved_both.append(name)
    elif item_ipopt["success"] == "True" and item_master["success"] == "False":
        solved_ipopt.append(name)
    elif item_ipopt["success"] == "False" and item_master["success"] == "False":
        solved_none.append(name)
    elif item_ipopt["success"] == "False" and item_master["success"] == "True":
        solved_master.append(name)
    else:
        raise ValueError("success not False/True in ", name)


def rel_error(master, ipopt, eps):
    minimum = min(master, ipopt)
    rel_errs = [(i - minimum) / (abs(minimum) + 1) for i in [master, ipopt]]
    if rel_errs[0] == 0:  # master was better
        if rel_errs[1] <= eps:
            return 2
        else:
            return 3
    elif rel_errs[1] == 0:  # Ipopt was better
        if rel_errs[0] <= eps:
            return 2
        else:
            return 1
    else:
        raise ValueError("BLA")


def solved_f_vs_iter():
    eps_f = 1e-4
    eps_iter = 1

    # rows = iterations
    # cols = funciton values
    dct = {
        1: {1: 0, 2: 0, 3: 0},
        2: {1: 0, 2: 0, 3: 0},
        3: {1: 0, 2: 0, 3: 0},
    }

    names = []
    for name in solved_both:
        ipopt = csvdict_ipopt[name]
        master = csvdict_master[name]
        f_ipopt = float(ipopt["f"])
        f_master = float(master["f"])
        i_ipopt = int(ipopt["iter"])
        i_master = int(master["iter"])
        rel_err_iter = rel_error(i_master, i_ipopt, eps_iter)
        rel_err_f = rel_error(f_master, f_ipopt, eps_f)
        dct[rel_err_iter][rel_err_f] += 1
        if rel_err_f == 2:
            names.append(name)
            # print(name, rel_err_iter, rel_err_f)
    return names, dct


def get_iter_com_prob_to_solve(itero):
    im = [(i, itero.count(i)) for i in sorted(list(set(itero)))]
    s = 0
    results = []
    for im_i, im_c in im:
        s += im_c
        results.append((im_i, s / len(itero)))
    x, y = zip(*results)
    return x, y

names, dct = solved_f_vs_iter()

iter_master = [int(v["iter"]) for k, v in csvdict_master.items() if k in names]
iter_ipopt = [int(v["iter"]) for k, v in csvdict_ipopt.items() if k in names]
iter_worst = [i + random.randint(1, 100) for i in iter_master]

factor = [min(i,j) for i,j in zip(iter_master, iter_ipopt)]

iter_master_factor = sorted([m / f for m, f in zip(iter_master, factor)])
iter_ipopt_factor = sorted([m / f for m, f in zip(iter_ipopt, factor)])
iter_worst_factor = sorted([m / f for m, f in zip(iter_worst, factor)])

fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(1, 1, 1)

ax.semilogx(iter_master_factor, np.linspace(0, 1, len(names)),
            basex=2, linewidth='3', label="NOONTIME")
ax.semilogx(iter_ipopt_factor, np.linspace(0, 1, len(names)),
            basex=2, linewidth='3', alpha=0.7, label="IPOPT")
ax.grid(which='both', linestyle=':', linewidth='0.5', color='black')
ax.minorticks_on()
plt.xlabel("Relative Overhead: Number of Iterations / Optimal Iteration")
plt.ylabel("Relative Number of problems")
plt.legend()
plt.tight_layout()
plt.show()
