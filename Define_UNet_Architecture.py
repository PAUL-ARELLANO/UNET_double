# TRAINING THE UNET MODEL ===================================================
import tensorflow as tf
from tensorflow.keras import layers, Model

from osgeo import gdal
import rasterio
print(rasterio.__version__)
import numpy as np
from sklearn.model_selection import train_test_split

PATCH_SIZE = 256  # Choose size based on GPU memory

def create_patches(image_path, mask_path, patch_size):
    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        img = src_img.read()  # (11, H, W)
        mask = src_mask.read(1)  # (H, W)

    H, W = mask.shape
    patches_img = []
    patches_mask = []

    for i in range(0, H - patch_size, patch_size):  # Loop through image height
        for j in range(0, W - patch_size, patch_size):  # Loop through image width
            img_patch = img[:, i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]

            if img_patch.shape[1] == patch_size and img_patch.shape[2] == patch_size:
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)

    print(f"✅ Extracted {len(patches_img)} patches")
    return np.array(patches_img), np.array(patches_mask)

# Generate patches
X, Y = create_patches(
    "C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/U_net/TILE130_full/TILE130_full_11bands_renamed.tif",
    "C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/U_net/TILE130_full/tree_stress_mask_full.tif",
    256  # Patch size
)

print(f"Total patches extracted: {len(X)}")
print(f"X shape: {X.shape if isinstance(X, np.ndarray) else 'None'}")
print(f"Y shape: {Y.shape if isinstance(Y, np.ndarray) else 'None'}")

# Proceed only if there are patches
if len(X) > 0 and len(Y) > 0:
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
else:
    print("❌ Error: No patches were extracted. Check image/mask paths and patch extraction logic.")
    exit()

# Split into train, validation, test (80/10/10)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print(f"✅ Patches created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")


# DEFINE THE U-NET MODEL =====================================================================================


def unet_model(input_shape=(256, 256, 11), num_classes=5):
    inputs = layers.Input(input_shape)

    # Encoder (Downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder (Upsampling)
    u1 = layers.UpSampling2D((2, 2))(c3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)  # Softmax for multiclass

    model = Model(inputs, outputs)
    return model

# Create model
model = unet_model()
model.summary()



#2.  COMPILE THE MODEL
# Using Categorical Crossentropy since this is a multicalss segmentation

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#3. PREPARE YOUR DATA
# Since my Y-labeles are integer (0-4), I need to one-hot encode the labeles for multiclass training
from tensorflow.keras.utils import to_categorical

# Reshape the target labels into (num_samples, 256, 256, 1) first
Y_train = Y_train.reshape(-1, 256, 256, 1)
Y_val = Y_val.reshape(-1, 256, 256, 1)

# Convert to one-hot encoding (shape: (num_samples, 256, 256, 5))
Y_train = to_categorical(Y_train, num_classes=5)
Y_val = to_categorical(Y_val, num_classes=5)

# Before training, I need to transpose the input array to match the expected shape
# (N, 11, 256, 256) → (N, 256, 256, 11)
# This is because the model expects the last dimension to be the number of channels
# (11 in this case)
# I will do this for the training, validation, and test sets
# Note: This is specific to the U-Net model architecture
# If you are using a different model, you may not need to do this
# My data is currently in (batch, channels, height, width) format (which is PyTorch-style)
# TensorFlow/Keras expects data in (batch, height, width, channels) format. 
X_train = X_train.transpose(0, 2, 3, 1)  # Convert (N, 11, 256, 256) → (N, 256, 256, 11)
X_val = X_val.transpose(0, 2, 3, 1)
X_test = X_test.transpose(0, 2, 3, 1)



#4. TRAINING THE U-NET MODEL
# Batch size: Adjust based on GPU memory. Try 8 or 16 to start
# Epochs: Start with 50, then adjunst

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=50,
                    batch_size=16)


#5. EVALUATE THE MODEL
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc:.4f}")
