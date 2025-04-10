"""
idt_core.py - Core algorithms for Intensity Diffraction Tomography

This module implements the fundamental algorithms used in
Intensity Diffraction Tomography including forward models,
reconstruction methods, and related analysis tools.
"""

import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from idt_utils import angular_spectrum_propagation, get_transfer_function


class IDTForwardModel:
    """
    Forward model for Intensity Diffraction Tomography.
    
    This class handles the simulation of intensity measurements
    for a given sample under different illumination conditions.
    """
    
    def __init__(self, wavelength, pixel_size, sample_size=None):
        """
        Initialize the forward model.
        
        Parameters:
        -----------
        wavelength : float
            Illumination wavelength in meters
        pixel_size : float
            Size of each pixel in meters
        sample_size : tuple, optional
            Size of the sample grid (ny, nx)
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.sample_size = sample_size
        self.sample = None
        self.k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum
        
    def set_sample(self, sample):
        """
        Set the sample for simulation.
        
        Parameters:
        -----------
        sample : ndarray
            Complex transmission function of the sample
        """
        self.sample = sample
        if self.sample_size is None:
            self.sample_size = sample.shape
            
    def generate_incident_field(self, angle_x=0, angle_y=0):
        """
        Generate an incident plane wave field.
        
        Parameters:
        -----------
        angle_x, angle_y : float
            Incident angles in radians
            
        Returns:
        --------
        field : ndarray
            Complex incident field
        """
        if self.sample is None:
            raise ValueError("Sample must be set before generating fields")
            
        ny, nx = self.sample_size
        x = np.arange(-nx//2, nx//2) * self.pixel_size
        y = np.arange(-ny//2, ny//2) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        
        # Wave vector components
        kx = self.k0 * np.sin(angle_x)
        ky = self.k0 * np.sin(angle_y)
        
        # Plane wave
        field = np.exp(1j * (kx * X + ky * Y))
        
        return field
    
    def simulate_measurement(self, distance, incident_field=None, angles=None, 
                            return_complex=False):
        """
        Simulate intensity measurement at a specific distance.
        
        Parameters:
        -----------
        distance : float
            Propagation distance in meters
        incident_field : ndarray, optional
            Incident field (if not provided, uses normal incidence)
        angles : tuple, optional
            (angle_x, angle_y) for incidence angle if incident_field not provided
        return_complex : bool
            If True, returns complex field instead of intensity
            
        Returns:
        --------
        intensity : ndarray
            Simulated intensity pattern
        """
        if self.sample is None:
            raise ValueError("Sample must be set before simulation")
            
        # Generate incident field if not provided
        if incident_field is None:
            if angles is not None:
                angle_x, angle_y = angles
                incident_field = self.generate_incident_field(angle_x, angle_y)
            else:
                incident_field = self.generate_incident_field()
        
        # Field immediately after the sample
        field_after_sample = incident_field * self.sample
        
        # Propagate to the detector
        propagated_field = angular_spectrum_propagation(
            field_after_sample, 
            self.pixel_size, 
            self.wavelength, 
            distance
        )
        
        if return_complex:
            return propagated_field
        else:
            # Calculate intensity
            intensity = np.abs(propagated_field)**2
            return intensity
    
    def simulate_multi_distance(self, distances, incident_field=None, angles=None):
        """
        Simulate intensity measurements at multiple distances.
        
        Parameters:
        -----------
        distances : list or array
            List of propagation distances in meters
        incident_field : ndarray, optional
            Incident field (if not provided, uses normal incidence)
        angles : tuple, optional
            (angle_x, angle_y) for incidence angle if incident_field not provided
            
        Returns:
        --------
        intensities : list
            List of simulated intensity patterns
        """
        intensities = []
        
        for z in distances:
            intensity = self.simulate_measurement(z, incident_field, angles)
            intensities.append(intensity)
            
        return intensities
    
    def simulate_multi_angle(self, distance, max_angle, num_angles):
        """
        Simulate intensity measurements for multiple incident angles.
        
        Parameters:
        -----------
        distance : float
            Propagation distance in meters
        max_angle : float
            Maximum incident angle in radians
        num_angles : int
            Number of angles to simulate
            
        Returns:
        --------
        intensities : list
            List of simulated intensity patterns
        angles : list
            List of (angle_x, angle_y) pairs used for simulation
        """
        intensities = []
        angles = []
        
        # Generate a set of incident angles
        angle_range = np.linspace(-max_angle, max_angle, num_angles)
        
        for angle_x in angle_range:
            for angle_y in angle_range:
                intensity = self.simulate_measurement(distance, angles=(angle_x, angle_y))
                intensities.append(intensity)
                angles.append((angle_x, angle_y))
                
        return intensities, angles


class IDTReconstructor:
    """
    Reconstructor for Intensity Diffraction Tomography.
    
    This class implements various reconstruction methods
    for recovering the object's refractive index distribution
    from intensity measurements.
    """
    
    def __init__(self, wavelength, pixel_size):
        """
        Initialize the reconstructor.
        
        Parameters:
        -----------
        wavelength : float
            Illumination wavelength in meters
        pixel_size : float
            Size of each pixel in meters
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.k0 = 2 * np.pi / wavelength
        
    def transport_of_intensity(self, intensity1, intensity2, distance, regularization=1e-6):
        """
        Phase retrieval using the Transport of Intensity Equation (TIE).
        
        Parameters:
        -----------
        intensity1, intensity2 : ndarray
            Intensity measurements at two different distances
        distance : float
            Distance between the two measurement planes (z2 - z1)
        regularization : float
            Regularization parameter for the Laplacian inversion
            
        Returns:
        --------
        phase : ndarray
            Retrieved phase distribution
        """
        # Calculate intensity derivative (dI/dz)
        dIdz = (intensity2 - intensity1) / distance
        
        # FFT-based TIE solver
        ny, nx = intensity1.shape
        
        # Spatial frequencies
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, self.pixel_size))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, self.pixel_size))
        KX, KY = np.meshgrid(kx, ky)
        
        # Intensity and derivative in frequency domain
        I_fft = np.fft.fftshift(np.fft.fft2(intensity1))
        dIdz_fft = np.fft.fftshift(np.fft.fft2(dIdz))
        
        # TIE in frequency domain
        laplacian = -(KX**2 + KY**2)
        laplacian[0, 0] = 1  # Avoid division by zero
        laplacian_inv = 1 / (laplacian + regularization)
        laplacian_inv[0, 0] = 0  # Remove DC component
        
        # Solve for phase (in frequency domain)
        phase_fft = -self.k0 * laplacian_inv * dIdz_fft / (I_fft + 1e-10)
        
        # Transform back to spatial domain
        phase = np.fft.ifft2(np.fft.ifftshift(phase_fft)).real
        
        return phase
    
    def multi_distance_tie(self, intensities, distances, regularization=1e-6):
        """
        Multi-distance Transport of Intensity Equation (TIE) solver.
        
        Parameters:
        -----------
        intensities : list
            List of intensity measurements at different distances
        distances : list
            List of corresponding distances
        regularization : float
            Regularization parameter for the Laplacian inversion
            
        Returns:
        --------
        phase : ndarray
            Retrieved phase distribution
        """
        if len(intensities) < 3:
            raise ValueError("At least 3 intensity measurements are required")
            
        # Use central differences for better accuracy
        center_idx = len(distances) // 2
        center_intensity = intensities[center_idx]
        
        # Calculate intensity derivative using polynomial fitting
        z = np.array(distances)
        dIdz = np.zeros_like(center_intensity)
        
        for i in range(center_intensity.shape[0]):
            for j in range(center_intensity.shape[1]):
                intensity_values = [img[i, j] for img in intensities]
                # Fit polynomial and get derivative at center
                poly = np.polyfit(z, intensity_values, 2)
                dIdz[i, j] = np.polyval([2*poly[0], poly[1]], distances[center_idx])
                
        # FFT-based TIE solver
        ny, nx = center_intensity.shape
        
        # Spatial frequencies
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, self.pixel_size))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, self.pixel_size))
        KX, KY = np.meshgrid(kx, ky)
        
        # Intensity and derivative in frequency domain
        I_fft = np.fft.fftshift(np.fft.fft2(center_intensity))
        dIdz_fft = np.fft.fftshift(np.fft.fft2(dIdz))
        
        # TIE in frequency domain
        laplacian = -(KX**2 + KY**2)
        laplacian[0, 0] = 1  # Avoid division by zero
        laplacian_inv = 1 / (laplacian + regularization)
        laplacian_inv[0, 0] = 0  # Remove DC component
        
        # Solve for phase (in frequency domain)
        phase_fft = -self.k0 * laplacian_inv * dIdz_fft / (I_fft + 1e-10)
        
        # Transform back to spatial domain
        phase = np.fft.ifft2(np.fft.ifftshift(phase_fft)).real
        
        return phase
    
    def contrast_transfer_function(self, intensities, distances, regularization=1e-6):
        """
        Phase retrieval using Contrast Transfer Function (CTF) approach.
        
        Parameters:
        -----------
        intensities : list
            List of intensity measurements at different distances
        distances : list
            List of corresponding distances
        regularization : float
            Regularization parameter
            
        Returns:
        --------
        phase : ndarray
            Retrieved phase distribution
        """
        if len(intensities) < 2:
            raise ValueError("At least 2 intensity measurements are required")
            
        # Reference intensity (usually at z=0, but we'll use the first one)
        I0 = intensities[0]
        
        # Calculate normalized intensity contrast
        contrasts = [(I - I0) / I0 for I in intensities[1:]]
        z_values = distances[1:]
        
        # FFT-based CTF solution
        ny, nx = I0.shape
        
        # Spatial frequencies
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, self.pixel_size))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, self.pixel_size))
        KX, KY = np.meshgrid(kx, ky)
        KSQ = KX**2 + KY**2
        
        # Initialize arrays for weighted sum
        numerator = np.zeros((ny, nx), dtype=complex)
        denominator = np.zeros((ny, nx), dtype=float)
        
        for i, (contrast, z) in enumerate(zip(contrasts, z_values)):
            # Transform contrast to frequency domain
            contrast_fft = np.fft.fftshift(np.fft.fft2(contrast))
            
            # CTF is sinusoidal function of defocus
            # sin(π λ z |k|²)
            ctf = np.sin(np.pi * self.wavelength * z * KSQ)
            
            # Accumulate weighted contrasts
            numerator += contrast_fft * ctf
            denominator += ctf**2
        
        # Add regularization to avoid division by zero
        denominator += regularization
        
        # Solve for phase (in frequency domain)
        phase_fft = numerator / denominator
        
        # Constrain the solution to be real (phase only)
        phase_fft = 1j * np.imag(phase_fft)
        
        # Transform back to spatial domain
        phase = np.fft.ifft2(np.fft.ifftshift(phase_fft)).real
        
        return phase
    
    def first_born_approximation(self, intensity, reference_intensity, distance):
        """
        Reconstruct the object using the First Born Approximation.
        
        Parameters:
        -----------
        intensity : ndarray
            Measured intensity with object
        reference_intensity : ndarray
            Reference intensity without object (or incident intensity)
        distance : float
            Propagation distance
            
        Returns:
        --------
        reconstruction : ndarray
            Reconstructed object function
        """
        # Normalized intensity
        normalized = intensity / reference_intensity
        
        # Contrast (I - I0) / I0
        contrast = normalized - 1.0
        
        # FFT of contrast
        ny, nx = contrast.shape
        contrast_fft = np.fft.fftshift(np.fft.fft2(contrast))
        
        # Spatial frequencies
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, self.pixel_size))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, self.pixel_size))
        KX, KY = np.meshgrid(kx, ky)
        KSQ = KX**2 + KY**2
        
        # Inverse filter based on the Born approximation
        k = 2 * np.pi / self.wavelength
        kz = np.sqrt(k**2 - KSQ + 0j)
        
        # Calculate the transfer function
        H = 2 * np.sin(kz * distance / 2)
        
        # Avoid division by zero
        H[np.abs(H) < 1e-10] = 1e-10
        
        # Apply inverse filter
        object_fft = contrast_fft / H
        
        # Inverse FFT to get the object reconstruction
        reconstruction = np.fft.ifft2(np.fft.ifftshift(object_fft))
        
        return reconstruction
    
    def reconstruct_ri_profile(self, phase, thickness=None):
        """
        Convert phase to refractive index profile.
        
        Parameters:
        -----------
        phase : ndarray
            Retrieved phase distribution
        thickness : ndarray or float, optional
            Sample thickness map or constant thickness value in meters
            
        Returns:
        --------
        ri_profile : ndarray
            Refractive index profile of the sample
        """
        if thickness is None:
            raise ValueError("Thickness must be provided for RI reconstruction")
            
        # Calculate refractive index deviation from the phase
        delta_n = phase / (self.k0 * thickness)
        
        # Assuming a background refractive index of 1.0
        ri_profile = 1.0 + delta_n
        
        return ri_profile
    
    def phase_diversity(self, intensities, distances, max_iterations=50, regularization=1e-6):
        """
        Phase retrieval using phase diversity approach with iterative optimization.
        
        Parameters:
        -----------
        intensities : list
            List of intensity measurements at different distances
        distances : list
            List of corresponding distances
        max_iterations : int
            Maximum number of iterations
        regularization : float
            Regularization parameter
            
        Returns:
        --------
        phase : ndarray
            Retrieved phase distribution
        """
        if len(intensities) < 2:
            raise ValueError("At least 2 intensity measurements are required")
            
        # Initial guess from TIE
        phase = self.transport_of_intensity(intensities[0], intensities[1], 
                                           distances[1] - distances[0], 
                                           regularization)
        
        # Create a forward model for simulation
        ny, nx = intensities[0].shape
        forward_model = IDTForwardModel(self.wavelength, self.pixel_size, (ny, nx))
        
        # Assuming amplitude = sqrt(intensities[0])
        amplitude = np.sqrt(intensities[0])
        
        # Iterative refinement
        for iteration in range(max_iterations):
            # Create the complex field at the first plane
            field = amplitude * np.exp(1j * phase)
            forward_model.set_sample(field)
            
            # Calculate error metric
            error = 0
            for i, distance in enumerate(distances):
                # Skip the first plane which is our starting point
                if i == 0:
                    continue
                    
                # Calculate the propagated intensity
                delta_z = distance - distances[0]
                simulated = forward_model.simulate_measurement(delta_z)
                
                # Accumulate error
                error += np.sum((simulated - intensities[i])**2)
                
                # Update the phase using the difference
                intensity_ratio = np.sqrt(intensities[i] / (simulated + 1e-10))
                propagated_field = forward_model.simulate_measurement(delta_z, return_complex=True)
                back_propagated = angular_spectrum_propagation(
                    propagated_field * intensity_ratio,
                    self.pixel_size,
                    self.wavelength,
                    -delta_z
                )
                
                # Extract the updated phase
                new_phase = np.angle(back_propagated)
                
                # Blend with the current estimate
                alpha = 0.5  # Mixing parameter
                phase = (1 - alpha) * phase + alpha * new_phase
            
            # Check convergence (simplified)
            if iteration > 0 and abs(previous_error - error) / previous_error < 1e-4:
                break
                
            previous_error = error
            
        return phase
    
    def stochastic_optical_reconstruction_microscopy(self, intensities, psf, 
                                                  iterations=100, threshold=0.1):
        """
        Apply STORM-like reconstruction for super-resolution imaging.
        
        Parameters:
        -----------
        intensities : list
            List of intensity measurements (sparsely activated fluorophores)
        psf : ndarray
            Point spread function of the system
        iterations : int
            Number of iterations for reconstruction
        threshold : float
            Intensity threshold for localization
            
        Returns:
        --------
        high_res_image : ndarray
            Super-resolved image
        """
        # Get dimensions
        ny, nx = intensities[0].shape
        
        # Create a high-resolution grid (e.g., 5x upsampling)
        upsampling = 5
        high_res_image = np.zeros((ny*upsampling, nx*upsampling))
        
        # Process each frame
        for i, intensity in enumerate(intensities):
            # Apply threshold to find potential fluorophore locations
            peaks = intensity > (threshold * np.max(intensity))
            
            # Find local maxima
            labeled, num_features = ndimage.label(peaks)
            
            # For each potential fluorophore
            for j in range(1, num_features + 1):
                # Extract the region with this label
                region = (labeled == j)
                
                # Find the centroid of the region
                coords = np.where(region)
                y_center, x_center = np.mean(coords[0]), np.mean(coords[1])
                
                # Subpixel localization using center of mass
                window_size = 5
                ymin = max(0, int(y_center) - window_size)
                ymax = min(ny, int(y_center) + window_size + 1)
                xmin = max(0, int(x_center) - window_size)
                xmax = min(nx, int(x_center) + window_size + 1)
                
                # Extract region for refinement
                region_data = intensity[ymin:ymax, xmin:xmax]
                
                # Weighted centroid for subpixel localization
                y_coords, x_coords = np.meshgrid(np.arange(ymin, ymax), 
                                              np.arange(xmin, xmax), indexing='ij')
                weights = region_data - np.min(region_data)
                
                # Avoid division by zero
                if np.sum(weights) > 0:
                    y_refined = np.sum(y_coords * weights) / np.sum(weights)
                    x_refined = np.sum(x_coords * weights) / np.sum(weights)
                    
                    # Map to high-resolution grid
                    y_hr = int(y_refined * upsampling)
                    x_hr = int(x_refined * upsampling)
                    
                    # Make sure we're within bounds
                    if 0 <= y_hr < high_res_image.shape[0] and 0 <= x_hr < high_res_image.shape[1]:
                        # Add to the high-resolution image
                        high_res_image[y_hr, x_hr] += 1
        
        # Apply Gaussian blur to represent the localization precision
        high_res_image = ndimage.gaussian_filter(high_res_image, sigma=1.0)
        
        return high_res_image