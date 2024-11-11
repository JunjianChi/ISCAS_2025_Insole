import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define adjustable font sizes
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12

image_pattern = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
])


# Verify the shape of image_pattern
pattern_shape = image_pattern.shape  # Should be (33, 15)
print(f"Image pattern shape: {pattern_shape}")

# Define the file path (current directory)
file_path = os.path.join(os.getcwd(), 'data_output.csv')

# Read the CSV file
try:
    data = pd.read_csv(file_path)
    print("CSV file read successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Check if 'Matrix_0' column exists
if 'Data_Matrix_0' not in data.columns:
    print("Error: 'Matrix_0' column not found in the CSV file.")
    exit(1)

# Extract the 10th frame (index 9)
frame_index = 45  # Zero-based index
try:
    matrix_0_str = data.loc[frame_index, 'Data_Matrix_0']
    if pd.isna(matrix_0_str):
        print(f"Warning: 'Matrix_0' data is missing for frame {frame_index + 1}.")
        exit(1)
    print(f"Extracted 'Matrix_0' for frame {frame_index + 1}.")
except IndexError:
    print(f"Error: Frame index {frame_index} is out of range.")
    exit(1)
except Exception as e:
    print(f"Error extracting 'Matrix_0': {e}")
    exit(1)

# Convert the string to a list of numbers
try:
    # Assuming the string is comma-separated
    matrix_0_list = [float(x.strip()) for x in matrix_0_str.split(',')]
    print(f"Converted 'Matrix_0' to list with {len(matrix_0_list)} elements.")
except Exception as e:
    print(f"Error converting 'Matrix_0' to list: {e}")
    exit(1)

# Check if the number of elements matches the pattern
expected_num_elements = pattern_shape[0] * pattern_shape[1]  # 33 * 15 = 495
actual_num_elements = len(matrix_0_list)

if actual_num_elements != expected_num_elements:
    print(f"Error: Number of elements in 'Matrix_0' ({actual_num_elements}) does not match the expected number ({expected_num_elements}).")
    exit(1)

# Convert the list to a 2D NumPy array
reshaped_array = np.array(matrix_0_list).reshape(pattern_shape)
print(f"'Matrix_0' reshaped to {reshaped_array.shape} array.")

# Apply the image pattern to extract values where pattern == 1
masked_values = reshaped_array[image_pattern == 1]
print(f"Number of extracted values: {masked_values.size}")

# Verify if the number of extracted values is 253
if masked_values.size != 253:
    print(f"Warning: Extracted values count ({masked_values.size}) does not match expected count (253). Please check the image pattern.")
else:
    print("Successfully extracted 253 values.")

# Plot the histogram with adjustable font sizes
plt.figure(figsize=(10, 6))
plt.hist(masked_values, bins=30, edgecolor='black', color='skyblue')
plt.title('Histogram of Extracted Values', fontsize=TITLE_FONT_SIZE)
plt.xlabel('Value', fontsize=LABEL_FONT_SIZE)
plt.ylabel('Frequency', fontsize=LABEL_FONT_SIZE)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
