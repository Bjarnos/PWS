import json

with open("data/benchmarks.json", "r") as file:
    # format: ["ActivationFunction", "LossFunction", "Optimizer", LearningRate, TrainingTime, FinalAccuracy]
    benchmarks = json.load(file)

logs_template = r"""
# Benchmark analysis

## 1. Best combination per optimizer

These are the combinations with the highest final accuracy, per optimizer.<marker_1>

## 2. Optimal learning rate per optimizer

the learning rate per optimizer that is best on average, excluding MeanAbsoluteError because it lowers the average significantly.<marker_2>

## 3. All combinations from best to worst

This is a list of all combinations, sorted from highest to lowest accuracy.<marker_3>

"""

def item_to_string(item):
    return f"{item[2]} with {item[0]} and {item[1]} and a learning rate of {item[3]}"

# 1. Best combination per optimizer
logs = ""

combinations_by_optimizer = {
    "SGD": [],
    "SGDM": [],
    "AdaGrad": [],
    "RMSprop": [],
    "Adam": []
}

for item in benchmarks:
    combinations_by_optimizer[item[2]].append(item)

for optimizer, items in combinations_by_optimizer.items():
    highest = [None, None, None, None, None, 0]
    for item in items:
        if item[5] > highest[5]:
            if item[5] == highest[5]:
                raise RuntimeError("We have a tie!")

            highest = item

    logs += f"\n\nThe best combination for *{optimizer}* was {item_to_string(highest)}, with **{highest[5]*100}% accuracy**."

logs_template = logs_template.replace("<marker_1>", logs)

# 2. Optimal learning rate per combination
logs = ""

combinations_by_optimizer = {
    "SGD": [],
    "SGDM": [],
    "AdaGrad": [],
    "RMSprop": [],
    "Adam": []
}

for item in benchmarks:
    combinations_by_optimizer[item[2]].append(item)

accuracy: dict[str, dict[float, list[float]]] = {}

for optimizer, items in combinations_by_optimizer.items():
    accuracy[optimizer] = {}
    for item in items:
        if (item[1] == "MeanAbsoluteError"):
            continue
        if (accuracy[optimizer].get(item[3]) == None):
            accuracy[optimizer][item[3]] = []
        accuracy[optimizer][item[3]].append(item[5])

    max_acc = 0
    best_rate = -1
    for rate, accs in accuracy[optimizer].items():
        sum = 0
        for acc in accs:
            sum += acc
        avg = sum / len(accs)
        if (avg > max_acc):
            max_acc = avg
            best_rate = rate

    logs += f"\n\nThe best learning rate for *{optimizer}* was {best_rate}, with an average of **{max_acc*100:.4}%** accuracy."

logs_template = logs_template.replace("<marker_2>", logs)

# 3. All combinations from best to worst
logs = ""

sorted_benchmarks = list(benchmarks)
sorted_benchmarks.sort(key=lambda x: x[5], reverse=True)

for item in sorted_benchmarks:
    logs += f"\n\n{item_to_string(item)}, with **{item[5]*100:.2f}% accuracy**."

logs_template = logs_template.replace("<marker_3>", logs)

# Cleanup
with open("data/benchmark_analysis.md", "w", encoding="utf-8") as f:
    f.write(logs_template)
