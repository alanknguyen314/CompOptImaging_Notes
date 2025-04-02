# Fourier Optics for Computational Optical Imaging

This repository contains a comprehensive Python codebase that explains and demonstrates the fundamental concepts of Fourier Optics applied to Computational Optical Imaging. It serves as an educational resource for those learning these concepts from scratch.

## Overview

Fourier Optics is a powerful mathematical framework for understanding and analyzing optical systems using the principles of Fourier transforms. This codebase covers:

- Fourier transforms and their properties in optical contexts
- Spatial frequency domain analysis
- Convolution theorem and its applications in imaging
- Diffraction-based imaging
- Lensless imaging techniques

## Modules

### 1. Fourier Basics (`fourier_basics.py`)

This module introduces the fundamental mathematical concepts of Fourier transforms in optics, including:

- Basic Fourier transform properties
- Spatial frequency domain representation
- Linearity, shift theorem, and Parseval's theorem
- Analysis of spatial frequencies in images
- Nyquist-Shannon sampling theorem

### 2. Convolution Theorem (`convolution_theorem.py`)

This module explores the relationship between convolution in the spatial domain and multiplication in the frequency domain:

- Comparison of spatial and frequency domain convolution
- Edge detection using Fourier methods
- Optical Transfer Function (OTF) and Point Spread Function (PSF)
- Modulation Transfer Function (MTF) analysis

### 3. Diffraction Imaging (`diffraction_imaging.py`)

This module demonstrates diffraction-based imaging concepts:

- Fraunhofer (far-field) diffraction
- Fresnel (near-field) diffraction
- Airy disk and diffraction-limited imaging
- Resolution limits in optical systems
- Various aperture types (circular, rectangular, slits)
- Near-to-far field transition

### 4. Lensless Imaging (`lensless_imaging.py`)

This module covers computational techniques for imaging without conventional lenses:

- Digital holography
- Phase retrieval algorithms
- Coded aperture imaging
- Field propagation methods

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pillow (for certain pattern generation)

## Installation

```bash
pip install numpy scipy matplotlib pillow
```

## Usage

To run all demonstrations:

```bash
python main.py
```

To run a specific module's demonstrations:

```bash
python main.py --module basics      # Run Fourier basics demos
python main.py --module convolution # Run convolution theorem demos
python main.py --module diffraction # Run diffraction imaging demos
python main.py --module lensless    # Run lensless imaging demos
```

You can also import and use individual functions from each module:

```python
from fourier_basics import plot_image_and_spectrum
from convolution_theorem import demonstrate_convolution_theorem
from diffraction_imaging import demonstrate_fraunhofer_diffraction
from lensless_imaging import demonstrate_in_line_holography

# ... your code here ...
```

## Key Concepts

### Fourier Transform in Optics

The Fourier transform is used to decompose an image into its spatial frequency components, providing insights into the image's structure and facilitating various operations in the frequency domain.

### Convolution Theorem

The convolution theorem states that the Fourier transform of a convolution of two functions is equal to the product of their individual Fourier transforms:

F{f * g} = F{f} · F{g}

This principle allows efficient computation of convolution operations and is central to understanding how optical systems process images.

### Diffraction

Diffraction is the bending of waves around obstacles. In optics, it plays a crucial role in determining the resolution limits of imaging systems. The two main regimes are:

- **Fraunhofer diffraction**: Applies when the observation distance is much larger than the aperture size (far field)
- **Fresnel diffraction**: Applies when the observation distance is comparable to the aperture size (near field)

### Lensless Imaging

Lensless imaging techniques use computational methods to recover images without traditional lenses, often utilizing:

- Diffraction patterns
- Phase information
- Specially designed coded apertures
- Holographic principles

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This educational toolkit was developed to provide a comprehensive introduction to Fourier Optics concepts for Computational Optical Imaging. It draws inspiration from various textbooks and courses in the field, particularly:

- "Introduction to Fourier Optics" by Joseph W. Goodman
- "Principles of Optics" by Max Born and Emil Wolf
- "Computational Fourier Optics" by David Voelz

## Contributing

Contributions to improve and extend this educational toolkit are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.
