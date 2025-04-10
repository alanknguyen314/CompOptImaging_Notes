"""
idt_tomography.py - Tomographic reconstruction methods for 3D IDT

This module implements 3D reconstruction methods for IDT,
including filtered backprojection, diffraction tomography,
and other tomographic techniques for optical imaging.
"""

import numpy as np
from scipy import ndimage, fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from idt_utils import angular_spectrum_propagation, get_transfer_function
from idt_core import IDTForwardModel, IDTReconstructor


class DiffractionTomographyReconstructor:
    """
    Class for 3D diffraction tomography reconstruction.
    """
    
    def __init__(self, wavelength, pixel_size, n_background=1.0):
        """
        Initialize the diffraction tomography reconstructor.
        
        Parameters:
        -----------
        wavelength : float
            Illumination wavelength in meters
        pixel_size : float
            Size of each pixel in meters
        n_background : float
            Refractive index of the background medium
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.n_background = n_background
        self.k0 = 2 * np.pi / wavelength
        self.k_medium = self.k0 * n_background
        
    def create_ewald_sphere(self, nx, ny, nz, dkx, dky, dkz):
        """
        Create the Ewald sphere for mapping the spectrum.
        
        Parameters:
        -----------
        nx, ny, nz : int
            Number of points in each dimension
        dkx, dky, dkz : float
            Spatial frequency step sizes
            
        Returns:
        --------
        mask : ndarray
            Binary mask of the Ewald sphere
        """
        # Create 3D grid of spatial frequencies
        kx = (np.arange(nx) - nx//2) * dkx
        ky = (np.arange(ny) - ny//2) * dky
        kz = (np.arange(nz) - nz//2) * dkz
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate radial distance from origin
        KSQ = KX**2 + KY**2 + KZ**2
        
        # Create Ewald sphere mask
        # In diffraction tomography, measured projections are mapped
        # onto a spherical surface in Fourier space known as the Ewald sphere
        mask = np.abs(np.sqrt(KSQ) - self.k_medium) < (0.5 * dkz)
        
        return mask
    
    def filtered_backprojection(self, projections, angles, filter_type='ramp'):
        """
        Reconstruct a 3D volume using filtered backprojection.
        
        Parameters:
        -----------
        projections : list of ndarray
            List of 2D projection images
        angles : list or array
            Corresponding projection angles in radians
        filter_type : str
            Type of filter to apply ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann')
            
        Returns:
        --------
        volume : ndarray
            Reconstructed 3D volume
        """
        # Convert projections to numpy array
        projections = np.array(projections)
        num_angles = len(angles)
        
        # Get projection dimensions
        if len(projections.shape) == 3:
            num_slices, height, width = projections.shape
        else:
            height, width = projections.shape
            num_slices = 1
            projections = projections.reshape(1, height, width)
            
        # Initialize volume
        volume = np.zeros((height, height, height))
        
        # Create ramp filter in frequency domain
        padded_width = int(np.ceil(width * np.sqrt(2)))
        pad_width = padded_width - width
        
        # Prepare filter
        filter_func = self._create_filter(padded_width, filter_type)
        
        # Process each slice
        for i in range(num_slices):
            # Filtered projections for this slice
            filtered_projections = np.zeros((num_angles, padded_width))
            
            for j in range(num_angles):
                # Pad projection
                padded_proj = np.pad(projections[i, j], (0, pad_width), mode='constant')
                
                # FFT of the projection
                proj_fft = np.fft.fftshift(np.fft.fft(padded_proj))
                
                # Apply filter in frequency domain
                filtered_proj_fft = proj_fft * filter_func
                
                # Inverse FFT to get filtered projection
                filtered_proj = np.real(np.fft.ifft(np.fft.ifftshift(filtered_proj_fft)))
                
                # Store the filtered projection
                filtered_projections[j] = filtered_proj
            
            # Backproject to create the volume slice
            for j in range(num_angles):
                angle = angles[j]
                
                # Backproject the filtered projection
                self._backproject_slice(volume[:, :, i], filtered_projections[j, :width], angle)
                
        # Scale the volume
        volume *= np.pi / (2 * num_angles)
        
        return volume
    
    def _create_filter(self, size, filter_type):
        """
        Create a frequency filter for the filtered backprojection.
        
        Parameters:
        -----------
        size : int
            Size of the filter
        filter_type : str
            Type of filter to create
            
        Returns:
        --------
        filter_func : ndarray
            Filter function in frequency domain
        """
        # Create ramp filter - scaled to match FFT conventions
        n = np.arange(-size//2, size//2) / (size/2)
        filter_func = np.abs(n)
        
        # Apply additional window if specified
        if filter_type == 'shepp-logan':
            # Shepp-Logan filter
            filter_func[1:] *= np.sin(np.pi * n[1:]) / (np.pi * n[1:])
        elif filter_type == 'cosine':
            # Cosine filter
            filter_func *= np.cos(np.pi * n / 2)
        elif filter_type == 'hamming':
            # Hamming filter
            filter_func *= 0.54 + 0.46 * np.cos(np.pi * n)
        elif filter_type == 'hann':
            # Hann filter
            filter_func *= (1 + np.cos(np.pi * n)) / 2
            
        return filter_func
    
    def _backproject_slice(self, slice_2d, projection, angle):
        """
        Backproject a filtered projection onto a 2D slice.
        
        Parameters:
        -----------
        slice_2d : ndarray
            2D slice of the volume to update
        projection : ndarray
            1D filtered projection
        angle : float
            Projection angle in radians
        """
        height, width = slice_2d.shape
        
        # Create a meshgrid for the slice
        x = np.arange(width) - width/2
        y = np.arange(height) - height/2
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        
        # Map rotated coordinates to projection indices
        projection_indices = np.round(X_rot + projection.size/2).astype(int)
        
        # Clip indices to valid range
        valid_indices = (projection_indices >= 0) & (projection_indices < projection.size)
        
        # Update slice with backprojected values
        slice_2d[valid_indices] += projection[projection_indices[valid_indices]]
    
    def diffraction_tomography_reconstruction(self, projections, incidence_angles, 
                                           regularization=1e-6, padding=True):
        """
        Reconstruct a 3D volume using diffraction tomography principles.
        
        Parameters:
        -----------
        projections : list of ndarray
            Complex field measurements for different incidence angles
        incidence_angles : list of tuple
            List of (theta, phi) angles for each projection
        regularization : float
            Regularization parameter for inversion
        padding : bool
            Whether to use padding to reduce artifacts
            
        Returns:
        --------
        volume : ndarray
            Reconstructed 3D complex refractive index distribution
        """
        # Get projection dimensions
        num_projections = len(projections)
        ny, nx = projections[0].shape
        
        # Determine volume size (assuming cubic volume)
        nz = nx
        
        # Initialize 3D Fourier space
        if padding:
            pad_size = nx // 2
            volume_ft = np.zeros((nx+2*pad_size, ny+2*pad_size, nz+2*pad_size), dtype=complex)
            count_map = np.zeros((nx+2*pad_size, ny+2*pad_size, nz+2*pad_size))
        else:
            volume_ft = np.zeros((nx, ny, nz), dtype=complex)
            count_map = np.zeros((nx, ny, nz))
        
        # Spatial frequencies
        if padding:
            nx_pad, ny_pad, nz_pad = volume_ft.shape
            dkx = 2 * np.pi / (nx_pad * self.pixel_size)
            dky = 2 * np.pi / (ny_pad * self.pixel_size)
            dkz = 2 * np.pi / (nz_pad * self.pixel_size)
        else:
            dkx = 2 * np.pi / (nx * self.pixel_size)
            dky = 2 * np.pi / (ny * self.pixel_size)
            dkz = 2 * np.pi / (nz * self.pixel_size)
            
        # Create frequency grids
        kx = np.fft.fftshift(np.fft.fftfreq(volume_ft.shape[0], self.pixel_size)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(volume_ft.shape[1], self.pixel_size)) * 2 * np.pi
        kz = np.fft.fftshift(np.fft.fftfreq(volume_ft.shape[2], self.pixel_size)) * 2 * np.pi
        
        # Process each projection
        for i, (projection, angles) in enumerate(zip(projections, incidence_angles)):
            theta, phi = angles
            
            # 2D FFT of the projection
            if padding:
                projection_padded = np.pad(projection, ((pad_size, pad_size), (pad_size, pad_size)), 
                                         mode='constant')
                proj_ft = np.fft.fftshift(np.fft.fft2(projection_padded))
            else:
                proj_ft = np.fft.fftshift(np.fft.fft2(projection))
            
            # Calculate incident wave vector
            kix = self.k_medium * np.sin(theta) * np.cos(phi)
            kiy = self.k_medium * np.sin(theta) * np.sin(phi)
            kiz = self.k_medium * np.cos(theta)
            
            # Map each projection to a slice of the Ewald sphere
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            
            # Calculate kz using Ewald sphere mapping
            KZ_sq = self.k_medium**2 - (KX - kix)**2 - (KY - kiy)**2
            
            # Filter out evanescent waves
            valid_indices = KZ_sq > 0
            KZ = np.zeros_like(KZ_sq)
            KZ[valid_indices] = kiz + np.sqrt(KZ_sq[valid_indices])
            
            # Convert to indices in the volume
            KZ_indices = np.round((KZ - kz[0]) / dkz).astype(int)
            
            # Filter valid indices
            valid_indices = (valid_indices & 
                           (KZ_indices >= 0) & 
                           (KZ_indices < volume_ft.shape[2]))
            
            # Map projection to 3D Fourier space
            for ix in range(volume_ft.shape[0]):
                for iy in range(volume_ft.shape[1]):
                    if valid_indices[ix, iy]:
                        iz = KZ_indices[ix, iy]
                        volume_ft[ix, iy, iz] += proj_ft[ix, iy]
                        count_map[ix, iy, iz] += 1
            
        # Normalize by the count map to handle overlapping contributions
        valid_counts = count_map > 0
        volume_ft[valid_counts] /= count_map[valid_counts]
        
        # Add regularization to empty voxels
        volume_ft[~valid_counts] = regularization
        
        # Inverse 3D FFT to get the volume
        volume = np.fft.ifftn(np.fft.ifftshift(volume_ft))
        
        # Extract the object region if padded
        if padding:
            volume = volume[pad_size:pad_size+nx, pad_size:pad_size+ny, pad_size:pad_size+nz]
            
        return volume
    
    def iterative_diffraction_tomography(self, measurements, angles, forward_model,
                                      max_iterations=20, regularization=1e-4):
        """
        Iterative reconstruction for diffraction tomography.
        
        Parameters:
        -----------
        measurements : list of ndarray
            Intensity measurements for different angles
        angles : list of tuple
            List of (theta, phi) angles for each measurement
        forward_model : object
            Forward model for simulation
        max_iterations : int
            Maximum number of iterations
        regularization : float
            Regularization parameter
            
        Returns:
        --------
        volume : ndarray
            Reconstructed 3D volume
        """
        # Get dimensions
        ny, nx = measurements[0].shape
        nz = nx  # Assume cubic volume for simplicity
        
        # Initialize volume with zeros
        volume = np.zeros((nz, ny, nx), dtype=complex)
        
        # Create reconstructor for 2D phase retrieval
        reconstructor = IDTReconstructor(self.wavelength, self.pixel_size)
        
        # First pass: reconstruct initial volume using phase retrieval on each projection
        for i, (measurement, angle) in enumerate(zip(measurements, angles)):
            # Use a reference measurement (could be provided or simulated)
            # For simplicity, we'll use a flat reference here
            reference = np.ones_like(measurement)
            
            # Retrieve phase using one of the methods
            # For example, using first Born approximation
            phase = reconstructor.first_born_approximation(measurement, reference, 0.0)
            
            # Backproject the phase into the volume
            # This is a simplified approach - in practice, need proper backprojection
            theta, phi = angle
            rotation_matrix = self._get_rotation_matrix(theta, phi)
            self._insert_slice(volume, phase, rotation_matrix)
        
        # Iterative refinement
        for iteration in range(max_iterations):
            # For each angle, simulate the measurement and update the volume
            for i, (measurement, angle) in enumerate(zip(measurements, angles)):
                # Simulate the current estimate
                simulated = forward_model.simulate_from_volume(volume, angle)
                
                # Calculate the error
                error = measurement - simulated
                
                # Backproject the error to update the volume
                theta, phi = angle
                rotation_matrix = self._get_rotation_matrix(theta, phi)
                error_backprojection = self._backproject_error(error, rotation_matrix)
                
                # Update volume (with regularization)
                update_step = 0.5  # Step size
                volume += update_step * error_backprojection
                
                # Apply regularization (e.g., Total Variation)
                volume = self._apply_regularization(volume, regularization)
                
            # Optional: monitor convergence
            
        return volume
    
    def _get_rotation_matrix(self, theta, phi):
        """
        Calculate the rotation matrix for a given angle.
        
        Parameters:
        -----------
        theta, phi : float
            Rotation angles in radians
            
        Returns:
        --------
        rotation_matrix : ndarray
            3x3 rotation matrix
        """
        # Rotation around y-axis by theta
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Rotation around z-axis by phi
        R_z = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        return np.dot(R_z, R_y)
    
    def _insert_slice(self, volume, slice_data, rotation_matrix):
        """
        Insert a 2D slice into the 3D volume.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume to update
        slice_data : ndarray
            2D slice data
        rotation_matrix : ndarray
            Rotation matrix for slice insertion
        """
        # This is a simplified implementation
        # In practice, need interpolation for arbitrary angles
        
        nz, ny, nx = volume.shape
        slice_ny, slice_nx = slice_data.shape
        
        # Center of the volume
        cx, cy, cz = nx//2, ny//2, nz//2
        
        # For each point in the output volume
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Coordinates relative to the center
                    x = ix - cx
                    y = iy - cy
                    z = iz - cz
                    
                    # Apply inverse rotation
                    coords = np.dot(np.linalg.inv(rotation_matrix), np.array([x, y, z]))
                    
                    # If the point is in the slice plane (z=0 after rotation)
                    if np.abs(coords[2]) < 0.5:
                        # Map to slice coordinates
                        slice_x = int(coords[0] + slice_nx//2)
                        slice_y = int(coords[1] + slice_ny//2)
                        
                        # Check bounds
                        if 0 <= slice_x < slice_nx and 0 <= slice_y < slice_ny:
                            # Update volume
                            volume[iz, iy, ix] = slice_data[slice_y, slice_x]
    
    def _backproject_error(self, error, rotation_matrix):
        """
        Backproject an error slice into the volume.
        
        Parameters:
        -----------
        error : ndarray
            2D error slice
        rotation_matrix : ndarray
            Rotation matrix for backprojection
            
        Returns:
        --------
        error_volume : ndarray
            3D error volume
        """
        # Create volume of the same size as the error
        ny, nx = error.shape
        error_volume = np.zeros((nx, ny, nx), dtype=complex)
        
        # Backproject error into all slices (simplified)
        for iz in range(nx):
            error_volume[iz, :, :] = error
            
        # Apply rotation (simplified)
        # In practice, need proper 3D rotation with interpolation
        
        return error_volume
    
    def _apply_regularization(self, volume, alpha):
        """
        Apply regularization to the volume.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume to regularize
        alpha : float
            Regularization parameter
            
        Returns:
        --------
        regularized_volume : ndarray
            Regularized 3D volume
        """
        # Simple Tikhonov regularization
        volume_magnitude = np.abs(volume)
        max_val = np.max(volume_magnitude)
        
        # Threshold small values
        threshold = alpha * max_val
        mask = volume_magnitude < threshold
        volume[mask] = 0
        
        return volume
    
    def visualize_volume(self, volume, threshold=0.5, figsize=(12, 10)):
        """
        Visualize a 3D volume using slices.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume to visualize
        threshold : float
            Threshold for visualization (between 0 and 1)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib Figure
            Figure with visualizations
        """
        nz, ny, nx = volume.shape
        
        # Calculate slice positions
        z_slices = [int(nz * 0.25), int(nz * 0.5), int(nz * 0.75)]
        y_slices = [int(ny * 0.25), int(ny * 0.5), int(ny * 0.75)]
        x_slices = [int(nx * 0.25), int(nx * 0.5), int(nx * 0.75)]
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # Normalize volume for visualization
        volume_vis = np.abs(volume)
        volume_vis = (volume_vis - np.min(volume_vis)) / (np.max(volume_vis) - np.min(volume_vis))
        
        # Plot XY slices (Z constant)
        for i, z in enumerate(z_slices):
            im = axes[0, i].imshow(volume_vis[z, :, :], cmap='viridis')
            axes[0, i].set_title(f'XY Slice (Z={z})')
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Y')
            
        # Plot XZ slices (Y constant)
        for i, y in enumerate(y_slices):
            im = axes[1, i].imshow(volume_vis[:, y, :], cmap='viridis')
            axes[1, i].set_title(f'XZ Slice (Y={y})')
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Z')
            
        # Plot YZ slices (X constant)
        for i, x in enumerate(x_slices):
            im = axes[2, i].imshow(volume_vis[:, :, x], cmap='viridis')
            axes[2, i].set_title(f'YZ Slice (X={x})')
            axes[2, i].set_xlabel('Y')
            axes[2, i].set_ylabel('Z')
            
        plt.tight_layout()
        return fig