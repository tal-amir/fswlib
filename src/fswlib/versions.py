# src/fswlib/versions.py

import sys, torch, numpy
import fswlib

def main():
    print(f'Python: {sys.version.split()[0]}');
    print(f'NumPy: {numpy.__version__}');
    print(f'Torch: {torch.__version__}');
    print(f'CUDA available: {torch.cuda.is_available()}');
    print(f'CUDA version: {torch.version.cuda}')


if __name__ == "__main__":
    main()
