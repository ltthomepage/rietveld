import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your .csv file
file_path = 'atomic_scatter_factors.atl'

# Helper function to parse values to either float or leave as string if non-numeric
def parse_value(value):
    try:
        return float(value)
    except ValueError:
        return value.strip()  # Remove any extra whitespace for non-numeric strings

# Read the .atl file and parse the data into a list of dictionaries, skipping the header row
data = []
with open(file_path, encoding='utf-8') as file:
    next(file)  # Skip the header line if it exists
    
    for line in file:
        values = line.strip().split(',')
        if len(values) < 11:
            print(f"Warning: Skipping line with insufficient data: {values}")
            continue
        data.append({
            'atomic_number': parse_value(values[0]),
            'symbol': values[1].strip(),
            'a1': parse_value(values[2]),
            'b1': parse_value(values[3]),
            'a2': parse_value(values[4]),
            'b2': parse_value(values[5]),
            'a3': parse_value(values[6]),
            'b3': parse_value(values[7]),
            'a4': parse_value(values[8]),
            'b4': parse_value(values[9]),
            'c': parse_value(values[10])
        })

# Convert the data to a pandas DataFrame for easier handling
df = pd.DataFrame(data)


# Define a function to retrieve atom data by symbol or atomic number
def get_atom(symbol_or_number):
    if isinstance(symbol_or_number, str):
        atom_row = df[df['symbol'] == symbol_or_number]
    else:
        atom_row = df[df['atomic_number'] == symbol_or_number]
        
    if not atom_row.empty:
        return atom_row.iloc[0].to_dict()  # Convert the first matched row to a dictionary
    else:
        print(f"Error: Atom {symbol_or_number} not found.")
        return None

# Define a function to calculate the atomic scattering factor f(s)
def atomic_scatter_factor(atom, s):
    if atom is None:
        raise ValueError("Atom data is None, cannot calculate scattering factor.")
    c = atom['c']
    a_vals = [atom['a1'], atom['a2'], atom['a3'], atom['a4']]
    b_vals = [atom['b1'], atom['b2'], atom['b3'], atom['b4']]
    
    # Calculate f(s) using the formula
    f_s = c + sum(a * np.exp(-b * s**2) for a, b in zip(a_vals, b_vals))
    return f_s

# Get Fe atom data
fe_atom = get_atom('Fe')


x = np.linspace(0.00, 0.3, 100)
plt.plot(x, atomic_scatter_factor(fe_atom, x))
plt.show()
