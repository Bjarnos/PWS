from config import DEVICE
from evaluation import test_model


def train_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device=DEVICE,
):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        # Backpropagation
        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)

            print(
                f"loss: {loss.item():>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    epochs=5,
):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print("-" * 30)

        train_epoch(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
        )

        test_model(
            test_dataloader,
            model,
            loss_fn,
        )

    print("Done!")