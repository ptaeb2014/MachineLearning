import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import rasterio

# Function to load raster data from a folder
def load_rasters_from_folder(folder_path):
    tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]
    rasters = []
    raster_shapes = set()  # Set to store unique shapes of rasters
    print(tiff_files)
    for file in tiff_files:
        with rasterio.open(os.path.join(folder_path, file)) as src:
            raster_data = src.read(1)  # Assuming single-band rasters
            raster_shapes.add(raster_data.shape)  # Add shape to set
            rasters.append(raster_data)
    # Check if there's only one unique shape
    if len(raster_shapes) == 1:
        return np.array(rasters)
    else:
        print("Warning: Rasters have different shapes. Removing rasters with different shapes.")
        # Filter out rasters with different shapes
        target_shape = raster_shapes.pop()  # Get the target shape
        filtered_rasters = [raster for raster in rasters if raster.shape == target_shape]
        return np.array(filtered_rasters)

# Define your neural network model
def create_model(input_shape, output_shape):
    #input_shape = (5, 608, 319)
    print(input_shape)

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse') # Adjust loss function as per your requirement
    print(model)
    return model

# Load raster data from folder
input_folder = "output-Wave-Pickering"
output_folder = "output-Wave-Pickering"
input_rasters = load_rasters_from_folder(input_folder)
output_rasters = load_rasters_from_folder(output_folder)

# Assuming input and output rasters have the same dimensions
input_shape = input_rasters.shape
print(input_shape)
output_shape = output_rasters.shape
print(output_shape)

# Create and compile the model
model = create_model(input_shape, output_shape)

# Train the model
# input_rasters = input_rasters.reshape(1, 5, 608, 319)
# print(input_rasters)

model.fit(input_rasters, output_rasters, epochs=10, batch_size=32, validation_split=0.2)  # Adjust epochs and batch_size as needed

# Predict downsampled rasters
# Replace X_new with your new wave raster data
X_new = r'Kitts-waves/2perc_WaveHeight_Alt_1_2.tif'
predicted_raster = model.predict(X_new)