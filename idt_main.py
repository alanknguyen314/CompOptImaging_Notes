"""
idt_main.py - Main script for Intensity Diffraction Tomography 

This script provides a comprehensive command-line interface
to run different IDT simulations and reconstructions.

Usage:
    python idt_main.py --example wave_propagation
    python idt_main.py --example multi_distance_phase_retrieval
    python idt_main.py --custom --wavelength 532e-9 --object sphere --propagation 2e-3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Import IDT modules
from idt_utils import create_grid, angular_spectrum_propagation, create_phase_object, plot_field
from idt_core import IDTForwardModel, IDTReconstructor
from idt_tomography import DiffractionTomographyReconstructor
from idt_examples import (
    example_wave_propagation,
    example_phase_objects,
    example_transport_of_intensity,
    example_multi_distance_phase_retrieval,
    example_born_approximation,
    example_tomographic_reconstruction,
    example_multi_angle_idt,
    example_intensity_video,
    example_fourier_slice_theorem,
    run_all_examples
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Intensity Diffraction Tomography (IDT) Simulator')
    
    # Example selection
    parser.add_argument('--example', type=str, choices=[
        'wave_propagation',
        'phase_objects',
        'transport_of_intensity',
        'multi_distance_phase_retrieval',
        'born_approximation',
        'tomographic_reconstruction',
        'multi_angle_idt',
        'intensity_video',
        'fourier_slice_theorem',
        'all'
    ], help='Run a predefined example')
    
    # Custom simulation parameters
    parser.add_argument('--custom', action='store_true', help='Run a custom simulation')
    parser.add_argument('--wavelength', type=float, default=633e-9, 
                      help='Wavelength in meters (default: 633e-9)')
    parser.add_argument('--grid_size', type=int, default=256, 
                      help='Grid size (default: 256)')
    parser.add_argument('--physical_size', type=float, default=100e-6, 
                      help='Physical size in meters (default: 100e-6)')
    parser.add_argument('--object', type=str, choices=['sphere', 'cylinder', 'custom'], 
                      default='sphere', help='Object type (default: sphere)')
    parser.add_argument('--refractive_index', type=float, default=1.01, 
                      help='Object refractive index (default: 1.01)')
    parser.add_argument('--radius', type=float, default=20e-6, 
                      help='Object radius in meters (default: 20e-6)')
    parser.add_argument('--propagation', type=float, default=1e-3, 
                      help='Propagation distance in meters (default: 1e-3)')
    parser.add_argument('--reconstruction', type=str, 
                      choices=['tie', 'multi_tie', 'ctf', 'born'], 
                      help='Reconstruction method')
    parser.add_argument('--output', type=str, help='Output file prefix for saving results')
    
    return parser.parse_args()


def run_example(example_name):
    """Run a predefined example."""
    examples = {
        'wave_propagation': example_wave_propagation,
        'phase_objects': example_phase_objects,
        'transport_of_intensity': example_transport_of_intensity,
        'multi_distance_phase_retrieval': example_multi_distance_phase_retrieval,
        'born_approximation': example_born_approximation,
        'tomographic_reconstruction': example_tomographic_reconstruction,
        'multi_angle_idt': example_multi_angle_idt,
        'intensity_video': example_intensity_video,
        'fourier_slice_theorem': example_fourier_slice_theorem,
        'all': run_all_examples
    }
    
    if example_name in examples:
        print(f"Running example: {example_name}")
        examples[example_name]()
    else:
        print(f"Example '{example_name}' not found!")


def run_custom_simulation(args):
    """Run a custom simulation with the specified parameters."""
    print("Running custom simulation with the following parameters:")
    print(f"  Wavelength: {args.wavelength} m")
    print(f"  Grid size: {args.grid_size} pixels")
    print(f"  Physical size: {args.physical_size} m")
    print(f"  Object type: {args.object}")
    print(f"  Refractive index: {args.refractive_index}")
    print(f"  Radius: {args.radius} m")
    print(f"  Propagation distance: {args.propagation} m")
    
    # Create simulation grid
    N = args.grid_size
    L = args.physical_size
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N
    
    # Create phase object
    params = {
        'n_bg': 1.0,
        'n_obj': args.refractive_index,
        'radius': args.radius,
        'wavelength': args.wavelength
    }
    phase_obj = create_phase_object(X, Y, args.object, params)
    
    # Plot object
    fig_obj = plot_field(phase_obj, title=f"{args.object.capitalize()} Phase Object", 
                       plot_type='both')
    
    # Create forward model
    forward_model = IDTForwardModel(args.wavelength, dx)
    forward_model.set_sample(phase_obj)
    
    # Simulate measurement
    intensity = forward_model.simulate_measurement(args.propagation)
    
    # Plot intensity
    fig_int = plt.figure(figsize=(10, 8))
    plt.imshow(intensity, cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title(f"Intensity at z={args.propagation*1e3:.1f} mm")
    plt.tight_layout()
    
    # Perform reconstruction if requested
    if args.reconstruction:
        print(f"Performing reconstruction using method: {args.reconstruction}")
        
        # Create reconstructor
        reconstructor = IDTReconstructor(args.wavelength, dx)
        
        if args.reconstruction == 'tie':
            # Need two intensities for TIE
            z1 = args.propagation * 0.9
            z2 = args.propagation * 1.1
            intensity1 = forward_model.simulate_measurement(z1)
            intensity2 = forward_model.simulate_measurement(z2)
            
            # Retrieve phase
            retrieved_phase = reconstructor.transport_of_intensity(
                intensity1, intensity2, z2 - z1
            )
            
        elif args.reconstruction == 'multi_tie':
            # Need multiple distances for multi-TIE
            distances = np.linspace(
                args.propagation * 0.5,
                args.propagation * 1.5,
                5
            )
            intensities = []
            for z in distances:
                intensities.append(forward_model.simulate_measurement(z))
                
            # Retrieve phase
            retrieved_phase = reconstructor.multi_distance_tie(intensities, distances)
            
        elif args.reconstruction == 'ctf':
            # Need multiple distances for CTF
            distances = np.linspace(
                args.propagation * 0.5,
                args.propagation * 1.5,
                5
            )
            intensities = []
            for z in distances:
                intensities.append(forward_model.simulate_measurement(z))
                
            # Retrieve phase
            retrieved_phase = reconstructor.contrast_transfer_function(intensities, distances)
            
        elif args.reconstruction == 'born':
            # Need reference intensity for Born approximation
            reference_field = np.ones((N, N), dtype=complex)
            reference_intensity = np.abs(
                angular_spectrum_propagation(
                    reference_field, dx, args.wavelength, args.propagation
                )
            )**2
            
            # Retrieve object function
            reconstruction = reconstructor.first_born_approximation(
                intensity, reference_intensity, args.propagation
            )
            retrieved_phase = np.angle(reconstruction)
        
        # Plot the recovered phase
        fig_phase = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.angle(phase_obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Phase (rad)')
        plt.title('True Phase')
        
        plt.subplot(1, 2, 2)
        plt.imshow(retrieved_phase, cmap='twilight')
        plt.colorbar(label='Phase (rad)')
        plt.title(f'Retrieved Phase ({args.reconstruction.upper()})')
        
        plt.tight_layout()
    
    # Save results if requested
    if args.output:
        fig_obj.savefig(f"{args.output}_object.png", dpi=300)
        fig_int.savefig(f"{args.output}_intensity.png", dpi=300)
        
        if args.reconstruction:
            fig_phase.savefig(f"{args.output}_phase.png", dpi=300)
            
        print(f"Results saved with prefix: {args.output}")
    
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    if args.example:
        run_example(args.example)
    elif args.custom:
        run_custom_simulation(args)
    else:
        print("Please specify either an example to run or use --custom for a custom simulation.")
        print("Use --help for more information.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)