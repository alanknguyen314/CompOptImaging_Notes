"""
main.py - Main executable to run all Fourier Optics demonstrations

This script imports and executes all the demonstrations from the various
modules, providing a comprehensive overview of Fourier Optics concepts
for Computational Optical Imaging.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import modules
from fourier_basics import run_demonstrations as run_fourier_basics
from convolution_theorem import run_demonstrations as run_convolution_theorem
from diffraction_imaging import run_demonstrations as run_diffraction_imaging
from lensless_imaging import run_demonstrations as run_lensless_imaging

def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='Fourier Optics for Computational Optical Imaging')
    parser.add_argument('--module', type=str, choices=['all', 'basics', 'convolution', 'diffraction', 'lensless'],
                        default='all', help='Module to run demonstrations for')
    args = parser.parse_args()
    
    print("=" * 80)
    print(" FOURIER OPTICS FOR COMPUTATIONAL OPTICAL IMAGING ".center(80, "="))
    print("=" * 80)
    
    if args.module in ['all', 'basics']:
        print("\n" + "=" * 80)
        print(" FOURIER BASICS ".center(80, "="))
        print("=" * 80)
        run_fourier_basics()
    
    if args.module in ['all', 'convolution']:
        print("\n" + "=" * 80)
        print(" CONVOLUTION THEOREM ".center(80, "="))
        print("=" * 80)
        run_convolution_theorem()
    
    if args.module in ['all', 'diffraction']:
        print("\n" + "=" * 80)
        print(" DIFFRACTION IMAGING ".center(80, "="))
        print("=" * 80)
        run_diffraction_imaging()
    
    if args.module in ['all', 'lensless']:
        print("\n" + "=" * 80)
        print(" LENSLESS IMAGING ".center(80, "="))
        print("=" * 80)
        run_lensless_imaging()
    
    print("\n" + "=" * 80)
    print(" DEMONSTRATION COMPLETE ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()