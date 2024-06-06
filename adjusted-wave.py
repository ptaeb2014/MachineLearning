import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from skimage.transform import resize

def load_and_preprocess_raster(file_path, target_shape=(608, 320)):
    with rasterio.open(file_path) as src:
        # Assuming single-band raster
        raster_data = src.read(1)
        raster_data = preprocess_data(raster_data)
        # Resize to the target shape
        raster_data = resize(raster_data, target_shape, preserve_range=True)
        raster_data = raster_data.reshape(target_shape[0], target_shape[1], 1)  # Add channel dimension
        # print(raster_data.shape())
    return raster_data

def preprocess_data(data):
    finite_data = data[np.isfinite(data)]
    fill_value = np.mean(finite_data) if finite_data.size > 0 else 0
    data = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    data = np.clip(data, min_float32, max_float32)
    return data

def load_rasters_from_folder(folder_path):
    tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]
    rasters = []
    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        raster_data = load_and_preprocess_raster(file_path)
        rasters.append(raster_data)
    return np.stack(rasters) if rasters else np.array([])  # Stack into a single numpy array if not empty


def save_raster(output_path, data, transform, meta):
    # Update metadata for saving the raster
    meta.update({
        'dtype': 'float32',
        'count': 1
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)
        #dst.set_transform(transform)

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # layers.UpSampling2D((2, 2)),  # Upsamples to (1216, 640)
        # layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(1, (3, 3), activation='relu', padding='same')
    ])
    model.compile(
        # loss=losses.MeanAbsoluteError(),
        # loss=losses.MeanSquaredError(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),)
    return model

input_folder = 'input-FloodDepth-Pickering'
output_folder = 'output-Wave-Pickering'

# Load data
X = load_rasters_from_folder(input_folder)
Y = load_rasters_from_folder(output_folder)

if X.size > 0 and Y.size > 0:
    print("Data loaded successfully. Shapes:", X.shape, Y.shape)
    model = create_model(X.shape[1:])  # Create model using the shape of the input
    model.fit(X, Y, epochs=10, batch_size=1, validation_split=0.2)
else:
    print("Failed to load data. Check your folders and data.")

# Assuming 'model' is already trained and loaded
X_new_path = r'Kitts-waves/2perc_WaveHeight_Alt1_2.tif'
X_new = load_and_preprocess_raster(X_new_path)

# Model prediction
X_new_batch = np.expand_dims(X_new, axis=0)  # Add batch dimension
predicted_raster = model.predict(X_new_batch)
predicted_raster = predicted_raster.squeeze()  # Remove batch dimension

# Load original raster to fetch its transform and metadata for saving the predicted raster
with rasterio.open(X_new_path) as src:
    transform = src.transform
    meta = src.meta

# Save the predicted raster with the same geospatial metadata as the original
output_path = 'predicted_output.tif'
save_raster(output_path, predicted_raster.astype(np.float32), transform, meta)

