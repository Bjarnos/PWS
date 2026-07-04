
# type: ignore
# TODO: Store plot format to the class so we don't have to call "for i in range(len(benchmarks))" every time
from neural_network.ActivationFunctions import __all__ as activation_functions

import matplotlib.pyplot as mpl
from matplotlib.widgets import Button
import numpy as np
import json

class Plot:
    def __init__(self, benchmarks: np.ndarray):
        self.activations = [a for a in activation_functions if a != "ActivationFunction"]
        self.benchmarks = benchmarks
        self.activation_index = 0

        self.fig = mpl.figure(figsize=(8, 8), num="Benchmarks")

        self.update_plot()
        mpl.show()

    def update_plot(self):
        self.fig.clf()
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95, hspace=0.4, wspace=0.4)

        activation = self.activations[self.activation_index]

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

        self.fig.suptitle(activation, fontsize=14, fontweight='bold')

        if not dictionary:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No benchmark data found for '{activation}'", ha='center', va='center', fontsize=12)
            ax.axis('off')

            self.draw()
            return
        
        fontsize = 6

        index = 0
        num_loss = len(dictionary)
        num_opt = len(dictionary[loss])
        for i, loss in enumerate(dictionary):
            for j, opt in enumerate(dictionary[loss]):
                index += 1
                data = dictionary[loss][opt]

                ax = self.fig.add_subplot(num_loss, num_opt, index)
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

        self.draw()

    def draw(self):
        ax_prev = self.fig.add_axes([0.35, 0.03, 0.12, 0.05])
        ax_next = self.fig.add_axes([0.53, 0.03, 0.12, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.previous)
        self.btn_next.on_clicked(self.next)

        self.fig.canvas.draw_idle()

    def previous(self, event):
        self.activation_index = (self.activation_index - 1) % len(self.activations)
        self.update_plot()

    def next(self, event):
        self.activation_index = (self.activation_index + 1) % len(self.activations)
        self.update_plot()

with open("benchmarks.json", "r") as file:
    benchmarks = np.asarray(json.load(file))

Plot(benchmarks)
