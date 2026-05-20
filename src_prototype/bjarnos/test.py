"""Test.py
Een scriptje om te kijken of PyTorch werkt.
"""

import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())  # False want hij is geinstalleerd als CPU only
