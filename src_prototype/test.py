"""Test.py
A little script to test if PyTorch works.
"""

import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())  # Likely False because we both don't own a GPU
