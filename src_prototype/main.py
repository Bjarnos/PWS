"""Main.py
The source code of our Neural Network prototype

Source of information: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

usage: main.py [args]

Neural Network Prototype

options:
  -h, --help            show this help message and exit
  --save-name SAVE_NAME
                        Filename to save the final model to.
  --load-name LOAD_NAME
                        Filename to load a pre-trained model from.
  -t, --type {train,test,interactive,tt,ti}
                        What the script should do. (tt = train+test, ti = train+interactive)
  --grid-size GRID_SIZE
                        Amount of images (rooted) to show to the Neural Network in the final test.
  -e, --epochs EPOCHS   Specifies the amount of training epochs.

# Example usages:
## Save backup for later testing:
python main.py --save-name backup --type train
## Test without training:
python main.py --load-name backup --type test
## Perform extensive testing
python main.py --grid-size 20 --epochs 10
## Test interatively:
python main.py --type ti
"""

# Import belangrijke libraries
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gui
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

if __name__ == "__main__":
    # Parse arguuments
    parser = argparse.ArgumentParser(description="Neural Network Prototype")
    parser.add_argument(
        "--save-name",
        help="Filename to save the final model to."
        )
    
    parser.add_argument(
        "--load-name",
        help="Filename to load a pre-trained model from."
        )
    
    parser.add_argument(
        "-t", "--type",
        choices=["train", "test", "interactive", "tt", "ti"], # tt = train+test, ti = train+interactive
        default="tt",
        help="What the script should do."
        )
    
    parser.add_argument(
        "--grid-size",
        default=5,
        type=int,
        help="Amount of images (rooted) to show to the Neural Network in the final test."
        )
    
    parser.add_argument(
        "-e", "--epochs",
        default=5,
        type=int,
        help="Specifies the amount of training epochs."
        )
    
    args = parser.parse_args()
    print(args)

    # Main code
    savename = args.save_name
    backupname = f"backups/model_{savename}.pth"
    if savename:
        dirname = os.path.dirname(__file__)
        backupfolder = os.path.join(dirname, "backups")
        if not os.path.exists(backupfolder):
            os.makedirs(backupfolder)
        else:
            if os.path.exists(os.path.join(dirname, backupname)):
                overwrite = input("Backup already exists! Do you want to overwrite (y/N)? ")
                if overwrite.strip().lower() == "y":
                    print("Continuing...")
                else:
                    print("Aborting.")
                    quit()

    loadname = args.load_name
    loadbackupname = f"backups/model_{loadname}.pth"
    if loadname:
        dirname = os.path.dirname(__file__)
        backupfolder = os.path.join(dirname, "backups")
        if not os.path.exists(backupfolder) or not os.path.exists(os.path.join(dirname, loadbackupname)):
            print("Backup doesn't exist!")
            quit()

    # Download training & test data from open datasets.
    # The data is automatically converted to a PyTorch compatible format
    # by the VisionDataset class of which MNIST inherits.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),    # input layer: 28 x 28 = 784 neurons
                nn.ReLU(),                  # activation function hidden layer: Rectified Linear Unit
                nn.Linear(512, 512),        # hidden layer: 512 neurons
                nn.ReLU(),                  # activation function output layer: Rectified Linear Unit
                nn.Linear(512, 10),         # output layer: 10 neurons
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    # We don't have a GPU to connect. All calculations will be done on our CPU
    device = "cpu"
    model = NeuralNetwork().to(device) # convert our config to an actual CPU Neural Network
    print(model)

    if loadname:
        model.load_state_dict(torch.load(loadbackupname, weights_only=True))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    if args.type in ["train", "tt", "ti"]:
        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        for t in range(args.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")
    
        if savename:
            # Save our model so we can continue training/using it later
            torch.save(model.state_dict(), backupname)
            print(f"Saved PyTorch Model State to {backupname}")
    
    if args.type in ["test", "tt"]:
        def draw_images(cols, rows):
            model.eval() # set the modus to testing rather than training
            
            num_correct = 0
            figure = plt.figure(figsize=(8, 8))
            for i in range(1, cols * rows + 1):
                sample_idx = torch.randint(len(test_data), size=(1,)).item()
                img, label = test_data[sample_idx]
                pred = model(img)[0].argmax(0)
                num_correct += int(pred == label)
                colorMap = "viridis" if pred == label else "inferno"
                
                figure.add_subplot(rows, cols, i)
                plt.title(f"{label} → {pred}") # 6, pred: 5
                plt.axis(False)
                plt.imshow(img.squeeze(), cmap=colorMap, interpolation="mitchell")
                
            print(f"correct: {num_correct}/{cols*rows} = {int(num_correct/(cols*rows)*100)}%")
            plt.show()

        test(test_dataloader, model, loss_fn)
        draw_images(args.grid_size, args.grid_size)

    if args.type in ["interactive", "ti"]:
        model.eval()
        timer = 0
        timer_dur = 10

        window = gui.Window()
        with torch.no_grad():
            while(window.running):
                window.update()
                timer += 1
                if (timer > timer_dur):
                    timer = 0
                    image = np.array([window.get_grid()])
                    logits = model(torch.from_numpy(image).float())[0]
                    window.prediction = torch.softmax(logits, dim=0)
                    print(window.prediction)
        
        window.quit()

            
