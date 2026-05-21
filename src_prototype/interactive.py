import numpy as np
import torch

import gui


def interactive_mode(model):
    model.eval()

    timer = 0
    timer_dur = 10

    window = gui.Window()

    with torch.no_grad():
        while window.running:
            window.update()

            timer += 1

            if timer > timer_dur:
                timer = 0

                image = np.array([
                    window.get_grid()
                ])

                logits = model(
                    torch.from_numpy(image).float()
                )[0]

                window.prediction = torch.softmax(
                    logits,
                    dim=0,
                )

    window.quit()