import os
import torch


def save_model(model, save_name):
    if not save_name:
        return

    dirname = os.path.dirname(__file__)

    backupfolder = os.path.join(
        dirname,
        "backups",
    )

    os.makedirs(
        backupfolder,
        exist_ok=True,
    )

    filename = os.path.join(
        backupfolder,
        f"model_{save_name}.pth",
    )

    if os.path.exists(filename):
        overwrite = input(
            "Backup already exists! "
            "Overwrite? (y/N): "
        )

        if overwrite.strip().lower() != "y":
            print("Aborting save.")
            return

    torch.save(
        model.state_dict(),
        filename,
    )

    print(f"Saved model to {filename}")


def load_model(model, load_name):
    dirname = os.path.dirname(__file__)

    filename = os.path.join(
        dirname,
        "backups",
        f"model_{load_name}.pth",
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Backup doesn't exist: {filename}"
        )

    model.load_state_dict(
        torch.load(
            filename,
            weights_only=True,
        )
    )

    print(f"Loaded model from {filename}")