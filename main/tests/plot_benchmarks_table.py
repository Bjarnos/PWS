
# type: ignore
# TODO: Store plot format to the class so we don't have to call "for i in range(len(benchmarks))" every time
from neural_network.ActivationFunctions import __all__ as activation_functions

import matplotlib.pyplot as mpl
from matplotlib.widgets import Button
import numpy as np
import json
import math

class Plot:
    def __init__(self, benchmarks: np.ndarray):
        self.activations = [a for a in activation_functions if a != "ActivationFunction"]
        self.benchmarks = benchmarks
        self.activation_index = 0
        self.cmap = mpl.get_cmap("Greys")

        self.dictionary: dict[str, dict[str, dict[str, dict[list[float]]]]] = {}
        self.fig = mpl.figure(figsize=(8, 8), num="Benchmarks")

        self.calculate_format()
        self.update_plot()
        mpl.show()

    def calculate_format(self):
        for i in range(len(benchmarks)):
            activation = str(benchmarks[i][0])
            loss = str(benchmarks[i][1])
            opt = str(benchmarks[i][2])
            rate, time, acc = benchmarks[i][3:]
            rate, time, acc = float(str(rate)), float(str(time)), float(str(acc))
            if (self.dictionary.get(opt) == None):
                self.dictionary[opt] = {}

            if (self.dictionary[opt].get(loss) == None):
                self.dictionary[opt][loss] = {}

            if (self.dictionary[opt][loss].get(activation) == None):
                self.dictionary[opt][loss][activation] = {'rate': [], 'time': [], 'acc': []}

            self.dictionary[opt][loss][activation]['rate'].append(rate)
            self.dictionary[opt][loss][activation]['time'].append(time)
            self.dictionary[opt][loss][activation]['acc'].append(acc)

        if not self.dictionary:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No benchmark data found", ha='center', va='center', fontsize=12)
            ax.axis('off')

            self.draw()
            return

    def update_plot(self):
        self.fig.clf()
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95, hspace=0.4, wspace=0.4)

        fontsize = 6

        self.fig.suptitle("Benchmark data per optimizer", fontsize=14, fontweight='bold')

        for idx, optimizer in enumerate(self.dictionary.keys()):
            num_loss = len(self.dictionary[optimizer])
            num_act = len(self.dictionary[optimizer]["MeanSquaredError"])
            data = np.zeros((num_loss, num_act))
            for i, loss in enumerate(self.dictionary[optimizer].keys()):
                for j, activation in enumerate(self.dictionary[optimizer][loss].keys()):
                    
                    acc = self.dictionary[optimizer][loss][activation]["acc"]
                    max_acc = max(acc)
                    print(1 - (max_acc))
                    data[(i, j)] = math.log(1 - max_acc) / 5 + 1

            ax = self.fig.add_subplot(3, 2, idx+1)
            ax.pcolor(data, cmap=self.cmap)
            ax.set_xticks(np.arange(0.5, num_act+0.5), self.dictionary[optimizer]["MeanSquaredError"].keys())
            ax.set_yticks(np.arange(0.5, num_loss+0.5), self.dictionary[optimizer].keys())
            ax.grid(False)
            ax.tick_params("x", labelrotation=270)

        self.fig.canvas.draw_idle()


with open("data/benchmarks.json", "r") as file:
    benchmarks = np.asarray(json.load(file))

Plot(benchmarks)
