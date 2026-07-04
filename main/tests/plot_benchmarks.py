import matplotlib.pyplot as mpl
import numpy as np
import json

def plot_benchmarks(benchmarks: np.ndarray, activation: str):

    dictionary: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for i in range(len(benchmarks)):
        if (benchmarks[i][0] == activation):
            loss = str(benchmarks[i][1])
            opt = str(benchmarks[i][2])
            rate, time, acc = benchmarks[i][3:]
            rate, time, acc = float(str(rate)), float(str(time)), float(str(acc))
            if (dictionary.get(loss) == None):
                dictionary[loss] = {}

            if (dictionary[loss].get(opt) == None):
                dictionary[loss][opt] = {'rate': [], 'time': [], 'acc': []}

            dictionary[loss][opt]['rate'].append(rate)
            dictionary[loss][opt]['time'].append(time)
            dictionary[loss][opt]['acc'].append(acc)
            
    mpl.figure(figsize=(8, 8), layout='tight', num=activation)
    fontsize = 6

    index = 0
    num_loss = len(dictionary)
    num_opt = len(dictionary[loss])
    for i, loss in enumerate(dictionary):
        for j, opt in enumerate(dictionary[loss]):
            index += 1
            data = dictionary[loss][opt]

            ax = mpl.subplot(num_loss, num_opt, index)
            ax.plot('rate', 'acc', 'b-', data=data)
            #mpl.plot('rate', 'time', 'r-', data=data)

            ax.grid(True, 'major', 'both', linewidth=1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_xscale('log')
            if opt in ['Adam', 'RMSprop']:
                learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
            else:
                learning_rates = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
            ax.set_xticks(learning_rates, learning_rates, fontsize=fontsize)

            if (i == num_loss-1):
                mpl.xlabel(opt)
            if (j == 0):
                mpl.ylabel(loss)

    mpl.show()




with open("benchmarks.json", "r") as file:
    benchmarks = np.asarray(json.load(file))

plot_benchmarks(benchmarks, "LeakyReLU")
