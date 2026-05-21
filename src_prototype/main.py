import argparse

from data import create_dataloaders
from model import (
    create_model,
    create_training_components,
)
from evaluation import (
    draw_images,
    test_model,
)
from train import train_model
from utils import (
    load_model,
    save_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Network Prototype"
    )

    parser.add_argument(
        "--save-name",
        help="Filename to save the final model to.",
    )

    parser.add_argument(
        "--load-name",
        help="Filename to load a pre-trained model from.",
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=[
            "train",
            "test",
            "interactive",
            "tt",
            "ti",
        ],
        default="tt",
        help="What the script should do.",
    )

    parser.add_argument(
        "--grid-size",
        default=5,
        type=int,
        help=(
            "Amount of images (rooted) "
            "to show to the Neural Network "
            "in the final test."
        ),
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        type=int,
        help="Specifies the amount of training epochs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    (
        training_data,
        test_data,
        train_dataloader,
        test_dataloader,
    ) = create_dataloaders()

    model = create_model()

    if args.load_name:
        load_model(model, args.load_name)

    loss_fn, optimizer = create_training_components(model)

    if args.type in ["train", "tt", "ti"]:
        train_model(
            model,
            train_dataloader,
            test_dataloader,
            loss_fn,
            optimizer,
            args.epochs,
        )

        if args.save_name:
            save_model(
                model,
                args.save_name,
            )

    if args.type in ["test", "tt"]:
        test_model(
            test_dataloader,
            model,
            loss_fn,
        )

        draw_images(
            model,
            test_data,
            args.grid_size,
            args.grid_size,
        )

    if args.type in ["interactive", "ti"]:
        from interactive import interactive_mode
        interactive_mode(model)


if __name__ == "__main__":
    main()