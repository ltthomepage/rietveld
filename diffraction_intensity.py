from atomic_factors import atomic_scatter_factor, get_atom
from auto_bragg import calculate_bragg_angles, CrystalSystem
import numpy as np
import matplotlib.pyplot as plt
from profile import PVf  # Import the profile function

def calculate_structure_factor(hkl, atom_locations, atoms, s):
    """
    Calculate the structure factor for a given reflection (hkl) and atom positions.
    """
    F = 0
    for i, (x, y, z) in enumerate(atom_locations):
        atom = atoms[i]
        f_j = atomic_scatter_factor(atom, s)  # Atomic scattering factor for the atom
        F += f_j * np.exp(2j * np.pi * (hkl[0] * x + hkl[1] * y + hkl[2] * z))  # Add atom contribution
    return F

def calculate_diffraction_intensity(hkl, atom_locations, atoms, s):
    """
    Calculate diffraction intensity for a given (hkl) reflection.
    """
    F = calculate_structure_factor(hkl, atom_locations, atoms, s)
    I = np.abs(F)**2  # Intensity is proportional to the square of the structure factor
    return I

# Define the element and atom locations
fe_atom = get_atom('Fe')
fe_atom_locations = [(0, 0, 0), (0.5, 0.5, 0.5)]  # Example for FCC structure, modify as needed
fe_a = 0.287  # Lattice constant in Å

# Set a threshold for diffraction intensity
intensity_threshold = 1e-5  # Set this value as appropriate for your case

# Maximum intensity to normalize to
max_intensity = 100

# Get Bragg angles for Fe (or another crystal system)
result = calculate_bragg_angles(fe_a, crystal_system=CrystalSystem.CUBIC)

# Initialize a list to store the intensities and angles for plotting
angles = []
intensities = []

# First pass: Calculate all intensities for normalization
raw_intensities = []

for d_hkl, two_theta, planes, num_planes in result:
    # 's' is the reciprocal of d_hkl
    s = 1 / d_hkl  # Calculate 's' as the reciprocal of 'd_hkl'
    
    # Calculate diffraction intensity for each (hkl) reflection
    intensity = calculate_diffraction_intensity(planes[-1], fe_atom_locations, [fe_atom] * len(fe_atom_locations), s)
    
    # Skip the reflection if the intensity is below the threshold
    if intensity < intensity_threshold:
        continue
    
    # Append raw intensity for normalization
    raw_intensities.append(intensity)

# Normalize using the maximum calculated intensity
max_calculated_intensity = max(raw_intensities) if raw_intensities else 1

# Second pass: Store scaled intensities and angles for plotting
for d_hkl, two_theta, planes, num_planes in result:
    s = 1 / d_hkl
    intensity = calculate_diffraction_intensity(planes[-1], fe_atom_locations, [fe_atom] * len(fe_atom_locations), s)
    
    if intensity < intensity_threshold:
        continue
    
    # Scale the intensity according to the maximum intensity
    scaled_intensity = (intensity / max_calculated_intensity) * max_intensity
    angles.append(two_theta)
    intensities.append(scaled_intensity)

# Plot the diffraction profile using the PVf function
plt.figure(figsize=(10, 6))
for angle, intensity in zip(angles, intensities):
    x = np.linspace(30, 110, 800)
    y = PVf(x,angle) * intensity
    plt.plot(x, y, color="blue")

# Formatting the plot
plt.xlabel("Diffraction Angle (2θ)")
plt.ylabel("Intensity (Arbitrary Units)")
plt.title("Diffraction Profile")
plt.show()
