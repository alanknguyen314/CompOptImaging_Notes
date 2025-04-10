"""
idt_examples.py - Practical examples of IDT techniques

This module provides practical examples and demonstrations
of various IDT methods for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

from idt_utils import (create_grid, angular_spectrum_propagation, fresnel_propagation,
                     create_phase_object, plot_field, get_transfer_function)
from idt_core import IDTForwardModel, IDTReconstructor
from idt_tomography import DiffractionTomographyReconstructor


def example_wave_propagation():
    """
    Demonstrate wave propagation using the angular spectrum method.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 512  # Number of grid points
    L = 200e-6  # Physical size (200 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a point source
    field = np.zeros((N, N), dtype=complex)
    field[N//2, N//2] = 1.0  # Point source at the center
    
    # Propagation distances
    distances = [1e-3, 2e-3, 5e-3, 10e-3]  # in meters
    
    # Plot results
    fig, axes = plt.subplots(2, len(distances), figsize=(16, 8))
    
    for i, distance in enumerate(distances):
        # Propagate using Angular Spectrum Method
        propagated_field = angular_spectrum_propagation(field, dx, wavelength, distance)
        
        # Intensity
        intensity = np.abs(propagated_field)**2
        
        # Plot intensity
        im1 = axes[0, i].imshow(intensity, cmap='viridis')
        axes[0, i].set_title(f'Intensity at z={distance*1000:.1f} mm')
        divider = make_axes_locatable(axes[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Plot phase
        phase = np.angle(propagated_field)
        im2 = axes[1, i].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[1, i].set_title(f'Phase at z={distance*1000:.1f} mm')
        divider = make_axes_locatable(axes[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    print("Notice how the wavefront evolves from a point source to a spherical wave.")
    print("This demonstrates the Huygens-Fresnel principle in wave optics.")


def example_phase_objects():
    """
    Demonstrate the creation and interaction of light with phase objects.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 512  # Number of grid points
    L = 200e-6  # Physical size (200 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create phase objects
    # Sphere
    sphere_params = {
        'n_bg': 1.0,
        'n_obj': 1.01,
        'radius': 20e-6,
        'wavelength': wavelength
    }
    sphere = create_phase_object(X, Y, 'sphere', sphere_params)
    
    # Cylinder
    cylinder_params = {
        'n_bg': 1.0,
        'n_obj': 1.02,
        'radius': 15e-6,
        'height': 30e-6,
        'wavelength': wavelength
    }
    cylinder = create_phase_object(X, Y, 'cylinder', cylinder_params)
    
    # Custom phase object (two spheres)
    def custom_phase_function(X, Y):
        # Two spheres
        r1 = np.sqrt((X - 30e-6)**2 + (Y - 30e-6)**2)
        r2 = np.sqrt((X + 30e-6)**2 + (Y + 30e-6)**2)
        
        z1 = np.zeros_like(r1)
        z2 = np.zeros_like(r2)
        
        radius1 = 20e-6
        radius2 = 15e-6
        
        mask1 = r1 <= radius1
        mask2 = r2 <= radius2
        
        z1[mask1] = 2 * np.sqrt(radius1**2 - r1[mask1]**2)
        z2[mask2] = 2 * np.sqrt(radius2**2 - r2[mask2]**2)
        
        delta_n = 0.01
        phase_change = k0 * delta_n * (z1 + z2)
        
        return np.exp(1j * phase_change)
    
    custom_params = {'phase_function': custom_phase_function}
    custom = create_phase_object(X, Y, 'custom', custom_params)
    
    # Create incident plane wave
    incident_field = np.ones((N, N), dtype=complex)
    
    # Interaction with objects
    field_sphere = incident_field * sphere
    field_cylinder = incident_field * cylinder
    field_custom = incident_field * custom
    
    # Propagation distance
    z = 5e-3  # 5 mm
    
    # Propagate fields
    propagated_sphere = angular_spectrum_propagation(field_sphere, dx, wavelength, z)
    propagated_cylinder = angular_spectrum_propagation(field_cylinder, dx, wavelength, z)
    propagated_custom = angular_spectrum_propagation(field_custom, dx, wavelength, z)
    
    # Plot results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Phase objects
    plot_phase = lambda field, ax, title: ax.imshow(np.angle(field), cmap='twilight', 
                                                 vmin=-np.pi, vmax=np.pi)
    plot_intensity = lambda field, ax, title: ax.imshow(np.abs(field)**2, cmap='viridis')
    
    # Original phase objects
    im1 = plot_phase(sphere, axes[0, 0], 'Sphere Phase')
    axes[0, 0].set_title('Sphere Phase')
    
    im2 = plot_phase(cylinder, axes[1, 0], 'Cylinder Phase')
    axes[1, 0].set_title('Cylinder Phase')
    
    im3 = plot_phase(custom, axes[2, 0], 'Custom Phase')
    axes[2, 0].set_title('Custom Phase')
    
    # Fields right after objects
    im4 = plot_intensity(field_sphere, axes[0, 1], 'Field after Sphere')
    axes[0, 1].set_title('Intensity after Sphere')
    
    im5 = plot_intensity(field_cylinder, axes[1, 1], 'Field after Cylinder')
    axes[1, 1].set_title('Intensity after Cylinder')
    
    im6 = plot_intensity(field_custom, axes[2, 1], 'Field after Custom')
    axes[2, 1].set_title('Intensity after Custom')
    
    # Propagated fields
    im7 = plot_intensity(propagated_sphere, axes[0, 2], 'Propagated Sphere')
    axes[0, 2].set_title(f'Intensity at z={z*1000:.1f}mm (Sphere)')
    
    im8 = plot_intensity(propagated_cylinder, axes[1, 2], 'Propagated Cylinder')
    axes[1, 2].set_title(f'Intensity at z={z*1000:.1f}mm (Cylinder)')
    
    im9 = plot_intensity(propagated_custom, axes[2, 2], 'Propagated Custom')
    axes[2, 2].set_title(f'Intensity at z={z*1000:.1f}mm (Custom)')
    
    # Add colorbars
    for i in range(3):
        for j in range(3):
            divider = make_axes_locatable(axes[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if j == 0:
                plt.colorbar(im1, cax=cax)
            else:
                plt.colorbar(im4, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    print("Notice how different phase objects create different intensity patterns at the detector.")
    print("This is the basis for phase contrast imaging in IDT.")


def example_transport_of_intensity():
    """
    Demonstrate phase retrieval using the Transport of Intensity Equation (TIE).
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 256  # Number of grid points
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a phase object (e.g., a sphere)
    params = {
        'n_bg': 1.0,
        'n_obj': 1.01,
        'radius': 15e-6,
        'wavelength': wavelength
    }
    phase_obj = create_phase_object(X, Y, 'sphere', params)
    
    # Original phase (ground truth)
    true_phase = np.angle(phase_obj)
    
    # Create incident plane wave
    incident_field = np.ones((N, N), dtype=complex)
    
    # Field immediately after the object
    field = incident_field * phase_obj
    
    # Propagation distances for TIE
    z1 = -10e-6  # Slightly before focus
    z2 = 10e-6   # Slightly after focus
    
    # Propagate fields
    field_z1 = angular_spectrum_propagation(field, dx, wavelength, z1)
    field_z2 = angular_spectrum_propagation(field, dx, wavelength, z2)
    
    # Calculate intensities
    intensity_z1 = np.abs(field_z1)**2
    intensity_z2 = np.abs(field_z2)**2
    
    # Create a reconstructor
    reconstructor = IDTReconstructor(wavelength, dx)
    
    # Retrieve phase using TIE
    retrieved_phase = reconstructor.transport_of_intensity(intensity_z1, intensity_z2, z2 - z1)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Intensities
    im1 = axes[0, 0].imshow(intensity_z1, cmap='viridis')
    axes[0, 0].set_title(f'Intensity at z={z1*1e6:.1f} µm')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    im2 = axes[0, 1].imshow(intensity_z2, cmap='viridis')
    axes[0, 1].set_title(f'Intensity at z={z2*1e6:.1f} µm')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Phases
    im3 = axes[1, 0].imshow(true_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('True Phase')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    im4 = axes[1, 1].imshow(retrieved_phase, cmap='twilight')
    axes[1, 1].set_title('Retrieved Phase (TIE)')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((retrieved_phase - true_phase)**2))
    # Normalize phases to account for constant offset
    true_phase_norm = true_phase - np.mean(true_phase)
    retrieved_phase_norm = retrieved_phase - np.mean(retrieved_phase)
    
    # Scale factor to account for amplitude differences
    scale = np.sum(true_phase_norm * retrieved_phase_norm) / np.sum(retrieved_phase_norm**2)
    retrieved_phase_scaled = retrieved_phase_norm * scale
    
    # Recalculate error after normalization and scaling
    rmse_norm = np.sqrt(np.mean((retrieved_phase_scaled - true_phase_norm)**2))
    
    print(f"Phase Retrieval Error (RMSE): {rmse:.6f}")
    print(f"Normalized Phase Retrieval Error: {rmse_norm:.6f}")
    print("The TIE method recovers the general shape of the phase but has limitations in accuracy.")
    print("It works best when the intensity varies slowly with propagation distance.")


def example_multi_distance_phase_retrieval():
    """
    Demonstrate phase retrieval using multi-distance measurements.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    
    # Create simulation grid
    N = 256  # Number of grid points
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a complex phase object (e.g., multiple spheres)
    def custom_phase_function(X, Y):
        # Three spheres of different sizes
        r1 = np.sqrt((X - 20e-6)**2 + (Y - 20e-6)**2)
        r2 = np.sqrt((X + 20e-6)**2 + (Y + 20e-6)**2)
        r3 = np.sqrt((X - 10e-6)**2 + (Y + 10e-6)**2)
        
        z = np.zeros_like(X)
        
        radius1 = 15e-6
        radius2 = 10e-6
        radius3 = 8e-6
        
        mask1 = r1 <= radius1
        mask2 = r2 <= radius2
        mask3 = r3 <= radius3
        
        z[mask1] += 2 * np.sqrt(radius1**2 - r1[mask1]**2)
        z[mask2] += 2 * np.sqrt(radius2**2 - r2[mask2]**2)
        z[mask3] += 2 * np.sqrt(radius3**2 - r3[mask3]**2)
        
        k0 = 2 * np.pi / wavelength
        delta_n = 0.01  # Refractive index difference
        phase_change = k0 * delta_n * z
        
        return np.exp(1j * phase_change)
    
    custom_params = {'phase_function': custom_phase_function}
    phase_obj = create_phase_object(X, Y, 'custom', custom_params)
    
    # Original phase (ground truth)
    true_phase = np.angle(phase_obj)
    
    # Create incident plane wave
    incident_field = np.ones((N, N), dtype=complex)
    
    # Field immediately after the object
    field = incident_field * phase_obj
    
    # Multiple propagation distances
    distances = [-50e-6, -25e-6, 0, 25e-6, 50e-6]
    
    # Propagate fields and calculate intensities
    intensities = []
    for z in distances:
        propagated = angular_spectrum_propagation(field, dx, wavelength, z)
        intensities.append(np.abs(propagated)**2)
    
    # Create a reconstructor
    reconstructor = IDTReconstructor(wavelength, dx)
    
    # Retrieve phase using different methods
    # 1. TIE with two distances
    tie_phase = reconstructor.transport_of_intensity(intensities[1], intensities[3], 
                                                  distances[3] - distances[1])
    
    # 2. Multi-distance TIE
    multi_tie_phase = reconstructor.multi_distance_tie(intensities, distances)
    
    # 3. Contrast Transfer Function (CTF)
    ctf_phase = reconstructor.contrast_transfer_function(intensities, distances)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sample intensities
    im1 = axes[0, 0].imshow(intensities[0], cmap='viridis')
    axes[0, 0].set_title(f'Intensity at z={distances[0]*1e6:.1f} µm')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    im2 = axes[0, 1].imshow(intensities[2], cmap='viridis')
    axes[0, 1].set_title(f'Intensity at z={distances[2]*1e6:.1f} µm')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    im3 = axes[0, 2].imshow(intensities[4], cmap='viridis')
    axes[0, 2].set_title(f'Intensity at z={distances[4]*1e6:.1f} µm')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    # True and retrieved phases
    im4 = axes[1, 0].imshow(true_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('True Phase')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax)
    
    im5 = axes[1, 1].imshow(tie_phase, cmap='twilight')
    axes[1, 1].set_title('TIE Phase Retrieval')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax)
    
    im6 = axes[1, 2].imshow(ctf_phase, cmap='twilight')
    axes[1, 2].set_title('CTF Phase Retrieval')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im6, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    # Plot multi-distance TIE separately
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(true_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title('True Phase')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    im2 = axes[1].imshow(multi_tie_phase, cmap='twilight')
    axes[1].set_title('Multi-distance TIE Phase Retrieval')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error metrics
    methods = ['Two-distance TIE', 'Multi-distance TIE', 'CTF']
    retrieved_phases = [tie_phase, multi_tie_phase, ctf_phase]
    
    print("Phase Retrieval Error Comparison:")
    for method, retrieved_phase in zip(methods, retrieved_phases):
        # Normalize phases to account for constant offset
        true_phase_norm = true_phase - np.mean(true_phase)
        retrieved_phase_norm = retrieved_phase - np.mean(retrieved_phase)
        
        # Scale factor to account for amplitude differences
        scale = np.sum(true_phase_norm * retrieved_phase_norm) / np.sum(retrieved_phase_norm**2)
        retrieved_phase_scaled = retrieved_phase_norm * scale
        
        # Calculate error after normalization and scaling
        rmse_norm = np.sqrt(np.mean((retrieved_phase_scaled - true_phase_norm)**2))
        
        print(f"{method} - Normalized RMSE: {rmse_norm:.6f}")
    
    print("\nMulti-distance methods generally perform better than two-distance methods.")
    print("The CTF approach can handle higher spatial frequencies but may be more sensitive to noise.")


def example_born_approximation():
    """
    Demonstrate object reconstruction using the First Born Approximation.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 256  # Number of grid points
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a weakly scattering object
    # For Born approximation to be valid, the phase shift should be small
    params = {
        'n_bg': 1.0,
        'n_obj': 1.005,  # Small refractive index contrast
        'radius': 15e-6,
        'wavelength': wavelength
    }
    phase_obj = create_phase_object(X, Y, 'sphere', params)
    
    # Create forward model
    forward_model = IDTForwardModel(wavelength, dx)
    forward_model.set_sample(phase_obj)
    
    # Propagation distance
    z = 1e-3  # 1 mm
    
    # Reference field (no object)
    reference_field = np.ones((N, N), dtype=complex)
    reference_intensity = np.abs(angular_spectrum_propagation(reference_field, dx, wavelength, z))**2
    
    # Simulate measurement with object
    measurement = forward_model.simulate_measurement(z)
    
    # Create reconstructor
    reconstructor = IDTReconstructor(wavelength, dx)
    
    # Reconstruct object using First Born Approximation
    reconstruction = reconstructor.first_born_approximation(measurement, reference_intensity, z)
    
    # True object function (proportional to refractive index contrast)
    true_obj = phase_obj - 1.0
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Intensities
    im1 = axes[0, 0].imshow(reference_intensity, cmap='viridis')
    axes[0, 0].set_title('Reference Intensity')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    im2 = axes[0, 1].imshow(measurement, cmap='viridis')
    axes[0, 1].set_title('Measurement with Object')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Object and reconstruction
    im3 = axes[1, 0].imshow(np.abs(true_obj), cmap='viridis')
    axes[1, 0].set_title('True Object (Magnitude)')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    im4 = axes[1, 1].imshow(np.abs(reconstruction), cmap='viridis')
    axes[1, 1].set_title('Reconstructed Object (Magnitude)')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    # Plot phase comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(np.angle(true_obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title('True Object (Phase)')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    im2 = axes[1].imshow(np.angle(reconstruction), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Reconstructed Object (Phase)')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    print("The First Born Approximation allows direct reconstruction of the object function.")
    print("It assumes weak scattering (small phase shifts) and single scattering events.")
    print("For stronger scattering, more advanced methods like multiple scattering models are needed.")


def example_tomographic_reconstruction():
    """
    Demonstrate 3D tomographic reconstruction from multiple projections.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 128  # Smaller for 3D simulation
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a 3D phantom (simple sphere)
    def create_3d_sphere(nx, ny, nz, radius, center=None):
        if center is None:
            center = [nx//2, ny//2, nz//2]
            
        phantom = np.zeros((nx, ny, nz), dtype=complex)
        
        x = np.arange(nx) - center[0]
        y = np.arange(ny) - center[1]
        z = np.arange(nz) - center[2]
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        R = np.sqrt(X**2 + Y**2 + Z**2)
        phantom[R <= radius] = 1.0 + 0.01j  # Small imaginary part for absorption
        
        return phantom
    
    # Create phantom
    nz = N
    phantom = create_3d_sphere(N, N, nz, N//8)
    
    # Simulate projections at different angles
    num_angles = 18
    angles = np.linspace(0, np.pi, num_angles)
    projections = []
    
    # Forward projection (simplified)
    for angle in angles:
        # Create rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Projection array
        projection = np.zeros((N, N), dtype=complex)
        
        # Simple projection (sum along rotated axis)
        for iz in range(nz):
            for iy in range(N):
                for ix in range(N):
                    # Coordinates relative to center
                    x = ix - N//2
                    y = iy - N//2
                    z = iz - nz//2
                    
                    # Rotated coordinates
                    x_rot = x * cos_a + z * sin_a
                    z_rot = -x * sin_a + z * cos_a
                    
                    # Map to projection coordinates
                    proj_x = int(x_rot + N//2)
                    proj_y = int(y + N//2)
                    
                    # Check bounds
                    if 0 <= proj_x < N and 0 <= proj_y < N:
                        # Accumulate value
                        projection[proj_y, proj_x] += phantom[ix, iy, iz]
        
        # Add to projections list
        projections.append(projection)
    
    # Create tomographic reconstructor
    tomography = DiffractionTomographyReconstructor(wavelength, dx)
    
    # Reconstruct using filtered backprojection
    volume = tomography.filtered_backprojection(
        [np.abs(proj) for proj in projections], 
        angles, 
        filter_type='ramp'
    )
    
    # Visualize the results
    fig = tomography.visualize_volume(volume)
    plt.suptitle('3D Tomographic Reconstruction', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Visualize ground truth for comparison
    fig = tomography.visualize_volume(np.abs(phantom))
    plt.suptitle('Ground Truth 3D Object', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("This example demonstrates the basic principle of tomographic reconstruction.")
    print("In real IDT, we would incorporate the effects of diffraction and use")
    print("more sophisticated methods like diffraction tomography algorithms.")


def example_multi_angle_idt():
    """
    Demonstrate IDT with multiple illumination angles.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 256  # Grid size
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a phase object
    params = {
        'n_bg': 1.0,
        'n_obj': 1.01,
        'radius': 15e-6,
        'wavelength': wavelength
    }
    phase_obj = create_phase_object(X, Y, 'sphere', params)
    
    # Create forward model
    forward_model = IDTForwardModel(wavelength, dx)
    forward_model.set_sample(phase_obj)
    
    # Propagation distance
    z = 1e-3  # 1 mm
    
    # Define illumination angles
    max_angle = 5 * np.pi / 180  # 5 degrees
    num_angles_1d = 5
    angles_1d = np.linspace(-max_angle, max_angle, num_angles_1d)
    
    # Simulate measurements with different illumination angles
    measurements = []
    angle_pairs = []
    
    for angle_x in angles_1d:
        for angle_y in angles_1d:
            # Generate incident field with angle
            incident_field = forward_model.generate_incident_field(angle_x, angle_y)
            
            # Simulate measurement
            intensity = forward_model.simulate_measurement(z, incident_field)
            
            # Add to collections
            measurements.append(intensity)
            angle_pairs.append((angle_x, angle_y))
    
    # For comparison, also get normal incidence
    normal_incidence = forward_model.simulate_measurement(z)
    
    # Demonstrate synthesis of higher NA image from multi-angle illumination
    # Simple coherent synthesis (sum complex fields)
    synthesized_field = np.zeros((N, N), dtype=complex)
    
    for i, angles in enumerate(angle_pairs):
        angle_x, angle_y = angles
        
        # Incident field
        incident_field = forward_model.generate_incident_field(angle_x, angle_y)
        
        # Complex field at detector
        field = forward_model.simulate_measurement(z, incident_field, return_complex=True)
        
        # Add to synthesized field with phase correction for incident angle
        kx = k0 * np.sin(angle_x)
        ky = k0 * np.sin(angle_y)
        phase_correction = np.exp(-1j * (kx * X + ky * Y))
        
        synthesized_field += field * phase_correction
    
    # Normalize
    synthesized_field /= len(angle_pairs)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Normal incidence
    im1 = axes[0, 0].imshow(normal_incidence, cmap='viridis')
    axes[0, 0].set_title('Normal Incidence')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # Some angled illuminations
    angles_to_show = [0, 12, 24]
    for i, idx in enumerate(angles_to_show):
        im = axes[0, i+1].imshow(measurements[idx], cmap='viridis')
        angle_x, angle_y = angle_pairs[idx]
        axes[0, i+1].set_title(f'Angle: ({angle_x*180/np.pi:.1f}°, {angle_y*180/np.pi:.1f}°)')
        divider = make_axes_locatable(axes[0, i+1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Synthesized intensity
    im4 = axes[1, 0].imshow(np.abs(synthesized_field)**2, cmap='viridis')
    axes[1, 0].set_title('Synthesized Intensity')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax)
    
    # Synthesized phase
    im5 = axes[1, 1].imshow(np.angle(synthesized_field), cmap='twilight', 
                         vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Synthesized Phase')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax)
    
    # True phase
    im6 = axes[1, 2].imshow(np.angle(phase_obj), cmap='twilight', 
                         vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('True Phase')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im6, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    print("Multi-angle illumination in IDT provides:")
    print("1. Improved resolution by synthetic aperture imaging")
    print("2. More complete information about the object's structure")
    print("3. Better phase recovery due to diversity in measurements")


def example_intensity_video():
    """
    Create an animation showing intensity patterns at different distances.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 256  # Number of grid points
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a phase object
    params = {
        'n_bg': 1.0,
        'n_obj': 1.01,
        'radius': 20e-6,
        'wavelength': wavelength
    }
    phase_obj = create_phase_object(X, Y, 'sphere', params)
    
    # Create incident plane wave
    incident_field = np.ones((N, N), dtype=complex)
    
    # Field immediately after the object
    field = incident_field * phase_obj
    
    # Propagation distances
    z_min = 0
    z_max = 5e-3  # 5 mm
    num_frames = 50
    distances = np.linspace(z_min, z_max, num_frames)
    
    # Pre-compute intensities
    intensities = []
    for z in distances:
        propagated = angular_spectrum_propagation(field, dx, wavelength, z)
        intensity = np.abs(propagated)**2
        intensities.append(intensity)
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initial intensity plot
    im = ax.imshow(intensities[0], cmap='viridis')
    title = ax.set_title(f'Intensity at z={distances[0]*1e3:.2f} mm')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Animation update function
    def update(frame):
        im.set_array(intensities[frame])
        title.set_text(f'Intensity at z={distances[frame]*1e3:.2f} mm')
        return [im, title]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
    
    plt.tight_layout()
    plt.show()
    
    print("The animation shows how the intensity pattern evolves with propagation distance.")
    print("This is the fundamental data used in IDT for phase retrieval and 3D reconstruction.")


def example_fourier_slice_theorem():
    """
    Demonstrate the Fourier Slice Theorem for diffraction tomography.
    """
    # Set parameters
    wavelength = 633e-9  # HeNe laser wavelength (meters)
    k0 = 2 * np.pi / wavelength  # Wavenumber
    
    # Create simulation grid
    N = 128  # Smaller for 3D simulation
    L = 100e-6  # Physical size (100 micrometers)
    X, Y, KX, KY = create_grid(N, L)
    dx = L / N  # Pixel size
    
    # Create a simple 2D phantom
    phantom = np.zeros((N, N), dtype=complex)
    radius = N // 6
    center_x, center_y = N // 2, N // 2
    for i in range(N):
        for j in range(N):
            r = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if r <= radius:
                phantom[i, j] = 1.0 + 0.01j  # Small imaginary part for absorption
    
    # Compute 2D Fourier transform of the phantom
    phantom_ft = np.fft.fftshift(np.fft.fft2(phantom))
    
    # Simulate projections at different angles
    num_angles = 4
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    projections = []
    projection_fts = []
    
    # Forward projection and demonstrate the Fourier Slice Theorem
    for angle in angles:
        projection = np.zeros(N, dtype=complex)
        
        # Projection along angle
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        for i in range(N):
            for j in range(N):
                # Coordinates relative to center
                x = i - N // 2
                y = j - N // 2
                
                # Rotated coordinates
                t = x * cos_a + y * sin_a
                
                # Map to projection array
                idx = int(t + N // 2)
                
                if 0 <= idx < N:
                    projection[idx] += phantom[i, j] / N
        
        projections.append(projection)
        
        # Compute Fourier transform of projection
        projection_ft = np.fft.fftshift(np.fft.fft(projection))
        projection_fts.append(projection_ft)
    
    # Plot results to illustrate the Fourier Slice Theorem
    fig = plt.figure(figsize=(15, 15))
    
    # Plot phantom and its Fourier transform
    ax1 = fig.add_subplot(3, 2, 1)
    im1 = ax1.imshow(np.abs(phantom), cmap='viridis')
    ax1.set_title('2D Phantom')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(3, 2, 2)
    im2 = ax2.imshow(np.log(1 + np.abs(phantom_ft)), cmap='viridis')
    ax2.set_title('2D Fourier Transform (log scale)')
    plt.colorbar(im2, ax=ax2)
    
    # Plot projections and their Fourier transforms for each angle
    for i, angle in enumerate(angles):
        row = i + 2
        
        # Plot projection
        ax = fig.add_subplot(3, 2, 2*row-1)
        x = np.arange(N) - N//2
        ax.plot(x, np.abs(projections[i]))
        ax.set_title(f'Projection at {angle*180/np.pi:.1f}°')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        
        # Plot 2D Fourier transform with the projection FT overlaid
        ax = fig.add_subplot(3, 2, 2*row)
        im = ax.imshow(np.log(1 + np.abs(phantom_ft)), cmap='viridis', alpha=0.7)
        plt.colorbar(im, ax=ax)
        
        # Overlay line showing the corresponding slice in Fourier space
        u = np.arange(N) - N//2
        x_line = u * np.cos(angle)
        y_line = u * np.sin(angle)
        
        # Plot the line through the Fourier transform
        ax.plot(y_line + N//2, x_line + N//2, 'r-', linewidth=2)
        
        # Plot the Fourier transform of the projection along the line
        ax.scatter(y_line + N//2, x_line + N//2, c=np.abs(projection_fts[i]), 
                cmap='cool', s=30, alpha=0.8)
        
        ax.set_title(f'Fourier Slice at {angle*180/np.pi:.1f}°')
    
    plt.tight_layout()
    plt.show()
    
    print("The Fourier Slice Theorem states that the 1D Fourier transform of a projection")
    print("is equal to a slice through the 2D Fourier transform of the object along the")
    print("same angle. This is a fundamental principle in tomographic reconstruction.")
    print("In diffraction tomography, this is extended to account for diffraction effects.")


def run_all_examples():
    """
    Run all examples in sequence.
    """
    examples = [
        example_wave_propagation,
        example_phase_objects,
        example_transport_of_intensity,
        example_multi_distance_phase_retrieval,
        example_born_approximation,
        example_tomographic_reconstruction,
        example_multi_angle_idt,
        example_intensity_video,
        example_fourier_slice_theorem
    ]
    
    for i, example in enumerate(examples):
        print(f"\n\n{'='*80}")
        print(f"Example {i+1}: {example.__name__}")
        print(f"{'='*80}\n")
        
        try:
            example()
        except Exception as e:
            print(f"Error running example: {str(e)}")
        
        # Wait a bit between examples
        if i < len(examples) - 1:
            time.sleep(1)


if __name__ == "__main__":
    # Uncomment to run all examples
    # run_all_examples()
    
    # Or run individual examples
    example_wave_propagation()
    # example_phase_objects()
    # example_transport_of_intensity()
    # example_multi_distance_phase_retrieval()
    # example_born_approximation()
    # example_tomographic_reconstruction()
    # example_multi_angle_idt()
    # example_intensity_video()
    # example_fourier_slice_theorem()