"""
idt_utils.py - Utility functions for Intensity Diffraction Tomography

This module provides fundamental functions for wave propagation, 
diffraction calculations, and related operations in IDT.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_grid(N, L):
    """
    Create a 2D spatial grid for simulation.
    
    Parameters:
    -----------
    N : int
        Number of pixels in each dimension
    L : float
        Physical size of the grid in meters
        
    Returns:
    --------
    x, y : ndarray
        Spatial coordinates
    kx, ky : ndarray
        Corresponding frequency coordinates
    """
    # Spatial coordinates
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Spatial frequencies
    dx = L / N
    dk = 2 * np.pi / L
    kx = np.fft.fftshift(np.fft.fftfreq(N, dx) * 2 * np.pi)
    ky = np.fft.fftshift(np.fft.fftfreq(N, dx) * 2 * np.pi)
    KX, KY = np.meshgrid(kx, ky)
    
    return X, Y, KX, KY


def angular_spectrum_propagation(field, dx, wavelength, distance, pad=True):
    """
    Propagate a complex field using the Angular Spectrum Method.
    
    Parameters:
    -----------
    field : ndarray
        Complex field at the initial plane
    dx : float
        Pixel size in meters
    wavelength : float
        Wavelength of light in meters
    distance : float
        Propagation distance in meters
    pad : bool
        Whether to pad the input field to avoid aliasing
        
    Returns:
    --------
    propagated_field : ndarray
        Complex field after propagation
    """
    if len(field.shape) != 2:
        raise ValueError("Field must be a 2D array")
    
    Ny, Nx = field.shape
    
    if pad:
        # Pad the field to at least 2x size to avoid aliasing
        pad_y = int(Ny)
        pad_x = int(Nx)
        field_padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    else:
        field_padded = field
    
    Ny_pad, Nx_pad = field_padded.shape
    
    # Calculate spatial frequencies
    k = 2 * np.pi / wavelength  # Wavenumber
    
    dfx = 1.0 / (Nx_pad * dx)
    dfy = 1.0 / (Ny_pad * dx)
    
    fx = np.fft.fftshift(np.fft.fftfreq(Nx_pad, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny_pad, dx))
    
    FX, FY = np.meshgrid(fx, fy)
    
    # Calculate transfer function
    fsq = FX**2 + FY**2
    transfer_function = np.exp(1j * k * distance * np.sqrt(1 - wavelength**2 * fsq))
    
    # Apply evanescent wave filtering
    transfer_function[fsq > 1.0/wavelength**2] = 0
    
    # Fourier transform of the input field
    FT_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_padded)))
    
    # Apply transfer function
    FT_propagated = FT_field * transfer_function
    
    # Inverse Fourier transform to get the propagated field
    propagated_field_padded = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(FT_propagated)))
    
    if pad:
        # Extract the original region
        propagated_field = propagated_field_padded[pad_y:pad_y+Ny, pad_x:pad_x+Nx]
    else:
        propagated_field = propagated_field_padded
    
    return propagated_field


def fresnel_propagation(field, dx, wavelength, distance):
    """
    Propagate a complex field using Fresnel diffraction.
    
    Parameters:
    -----------
    field : ndarray
        Complex field at the initial plane
    dx : float
        Pixel size in meters
    wavelength : float
        Wavelength of light in meters
    distance : float
        Propagation distance in meters
        
    Returns:
    --------
    propagated_field : ndarray
        Complex field after propagation
    """
    if len(field.shape) != 2:
        raise ValueError("Field must be a 2D array")
    
    Ny, Nx = field.shape
    k = 2 * np.pi / wavelength  # Wavenumber
    
    # Create coordinate grids
    x = np.arange(-Nx//2, Nx//2) * dx
    y = np.arange(-Ny//2, Ny//2) * dx
    X, Y = np.meshgrid(x, y)
    
    # Quadratic phase factor
    quad_phase = np.exp(1j * k / (2 * distance) * (X**2 + Y**2))
    
    # Apply first quadratic phase
    field = field * quad_phase
    
    # FFT
    field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    
    # Scaling factor
    scale = np.exp(1j * k * distance) / (1j * wavelength * distance)
    
    # New coordinates after propagation (frequency scaling)
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dx))
    FX, FY = np.meshgrid(fx, fy)
    
    # Second quadratic phase
    quad_phase_f = np.exp(1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    
    # Apply second quadratic phase and scaling
    propagated_field = scale * field_fft * quad_phase_f
    
    return propagated_field


def create_phase_object(X, Y, object_type='sphere', params=None):
    """
    Create a phase object for simulation.
    
    Parameters:
    -----------
    X, Y : ndarray
        Spatial coordinate grids
    object_type : str
        Type of object ('sphere', 'cylinder', 'custom')
    params : dict
        Object-specific parameters
        
    Returns:
    --------
    phase_obj : ndarray
        Complex transmission function of the phase object
    """
    if params is None:
        params = {}
    
    n_bg = params.get('n_bg', 1.0)  # Background refractive index
    n_obj = params.get('n_obj', 1.01)  # Object refractive index
    wavelength = params.get('wavelength', 633e-9)  # Wavelength in meters
    k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum
    
    phase_obj = np.ones_like(X, dtype=complex)
    
    if object_type == 'sphere':
        radius = params.get('radius', 10e-6)  # Radius in meters
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        thickness = params.get('thickness', 2 * radius)  # Default to diameter
        
        # Create spherical phase object
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        z = np.zeros_like(r)
        
        # Compute thickness at each point (assuming a sphere)
        mask = r <= radius
        z[mask] = 2 * np.sqrt(radius**2 - r[mask]**2)
        
        # Calculate phase change
        delta_n = n_obj - n_bg
        phase_change = k0 * delta_n * z
        
        # Create complex transmission function
        phase_obj = np.exp(1j * phase_change)
        
    elif object_type == 'cylinder':
        radius = params.get('radius', 10e-6)
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        height = params.get('height', 20e-6)
        
        # Create cylindrical phase object
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        z = np.zeros_like(r)
        
        # Set uniform thickness within the cylinder
        mask = r <= radius
        z[mask] = height
        
        # Calculate phase change
        delta_n = n_obj - n_bg
        phase_change = k0 * delta_n * z
        
        # Create complex transmission function
        phase_obj = np.exp(1j * phase_change)
        
    elif object_type == 'custom':
        # Custom phase function provided by the user
        phase_function = params.get('phase_function', None)
        if phase_function is not None:
            phase_obj = phase_function(X, Y)
    
    return phase_obj


def plot_field(field, title=None, vmin=None, vmax=None, cmap='viridis', 
               plot_type='intensity', phase_cmap='twilight', figsize=(10, 8)):
    """
    Plot a complex field (intensity, phase, or both).
    
    Parameters:
    -----------
    field : ndarray
        Complex field to plot
    title : str
        Plot title
    vmin, vmax : float
        Range for colormap
    cmap : str
        Colormap for intensity
    plot_type : str
        'intensity', 'phase', or 'both'
    phase_cmap : str
        Colormap for phase
    figsize : tuple
        Figure size
    """
    intensity = np.abs(field)**2
    phase = np.angle(field)
    
    if plot_type == 'intensity':
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(intensity, cmap=cmap, vmin=vmin, vmax=vmax)
        if title:
            ax.set_title(title)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
    elif plot_type == 'phase':
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(phase, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        if title:
            ax.set_title(title)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
    elif plot_type == 'both':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Intensity plot
        im1 = axes[0].imshow(intensity, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Intensity')
        axes[0].set_xlabel('x (pixels)')
        axes[0].set_ylabel('y (pixels)')
        
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Phase plot
        im2 = axes[1].imshow(phase, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase')
        axes[1].set_xlabel('x (pixels)')
        
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
    plt.tight_layout()
    return fig


def get_transfer_function(KX, KY, wavelength, z):
    """
    Calculate the transfer function for angular spectrum propagation.
    
    Parameters:
    -----------
    KX, KY : ndarray
        Spatial frequency grids
    wavelength : float
        Wavelength of light in meters
    z : float
        Propagation distance in meters
        
    Returns:
    --------
    tf : ndarray
        Complex transfer function
    """
    k = 2 * np.pi / wavelength  # Wavenumber
    ksq = KX**2 + KY**2
    kz = np.sqrt(k**2 - ksq)
    
    # Apply evanescent wave filtering
    kz = np.real(kz) + 1j * np.abs(np.imag(kz))
    
    # Transfer function
    tf = np.exp(1j * kz * z)
    
    # Filter out evanescent waves
    tf[ksq > k**2] = 0
    
    return tf