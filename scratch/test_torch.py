import torch
from PyQt5.QtWidgets import QApplication
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
x = torch.rand(5, 3)
print(x)
