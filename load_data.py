import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your XRD data file
file_path = 'test.txt'

# Load the data using pandas
# Assuming there is no header in the file; adjust as necessary
xrd_data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["angle", "intensity"])

# Plot the XRD data
plt.figure(figsize=(10, 6))
plt.plot(xrd_data['angle'], xrd_data['intensity'], color='blue', linewidth=1.5)

# Label the plot
plt.title('XRD Pattern')
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (a.u.)')

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.show()