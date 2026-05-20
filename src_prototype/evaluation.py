import matplotlib.pyplot as plt
import torch

from config import DEVICE


def test_model(
    dataloader,
    model,
    loss_fn,
    device=DEVICE,
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error:\n"
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}\n"
    )


def draw_images(
    model,
    test_data,
    cols=5,
    rows=5,
):
    model.eval()

    num_correct = 0

    figure = plt.figure(figsize=(8, 8))

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(
            len(test_data),
            size=(1,),
        ).item()

        img, label = test_data[sample_idx]

        pred = model(img)[0].argmax(0)

        num_correct += int(pred == label)

        color_map = (
            "viridis"
            if pred == label
            else "inferno"
        )

        figure.add_subplot(rows, cols, i)

        plt.title(f"{label} → {pred}")

        plt.axis(False)

        plt.imshow(
            img.squeeze(),
            cmap=color_map,
            interpolation="mitchell",
        )

    total = cols * rows

    print(
        f"correct: {num_correct}/{total} "
        f"= {int(num_correct / total * 100)}%"
    )

    plt.show()