import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import filedialog
from threading import Thread
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk, ImageColor
import random
# Account to CSC


opetus_ja_vastetiedot = pickle.load(open("opetustiedot.p", "rb"))
images = opetus_ja_vastetiedot[0]
masks = opetus_ja_vastetiedot[1]

# Scaling
images = images / 255
masks = masks / 255

# Adding a dimension
images = np.expand_dims(images, axis=3)
masks = np.expand_dims(masks, axis=3)

# CONFIG
image_shape = images[0].shape
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]
IMG_CHANNELS = image_shape[2]

# I really want to rewrite this.
train_ds = []
val_ds = []
for i in range(200):
    if np.random.rand() < 0.2:
        train_ds.append(i)
        if np.random.rand() < 0.0:
            masks[i, 50:60, 50:60] = 1
    else:
        val_ds.append(i)

traind_ds = np.array(train_ds)
val_ds = np.array(val_ds)

# Model def here


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Contracting Path
    c1 = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same')(p1)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same')(p2)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same')(p3)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = keras.layers.Conv2D(
        1024, (3, 3), activation='relu', padding='same')(p4)

    # Expansive Path
    u6 = keras.layers.Conv2DTranspose(
        512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.Concatenate()([u6, c4])
    c6 = keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same')(u6)

    u7 = keras.layers.Conv2DTranspose(
        256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.Concatenate()([u7, c3])
    c7 = keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same')(u7)

    u8 = keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.Concatenate()([u8, c2])
    c8 = keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same')(u8)

    u9 = keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.Concatenate()([u9, c1])
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)

    outputs = keras.layers.Conv2D(
        num_classes, (1, 1), activation="sigmoid")(c9)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_shape, num_classes=1)

# Model Config
epochs = 2

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "teras.weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    keras.callbacks.EarlyStopping(patience=5, verbose=1),
]
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    images[train_ds], masks[train_ds],
    batch_size=32
)

val_generator = val_datagen.flow(
    images[val_ds], masks[val_ds],
    batch_size=32
)


def train_model():
    global history
    teach_button.config(state=tk.DISABLED)
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
    )
    print("\nTraining finished.")
    teach_button.config(state=tk.DISABLED)


# UI code


image_label = None
mask_label = None
segmented_label = None
teach_button = None
root = None
history = None

model = make_model(input_shape=image_shape, num_classes=1)


def get_model_info():
    model_name = callbacks[0].filepath if callbacks and len(
        callbacks) > 0 else "undefined"
    return f"Current Model: {model_name}"


def teach_model_thread():
    # Train the model in a separate thread
    training_thread = Thread(target=train_model)
    training_thread.start()


def show_graph_button_click():
    if history is not None:
        # Plot training and validation metrics
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
    else:
        print("No training history available. Please train the model first.")


def show_image_button_click():
    global root, image_label, mask_label, segmented_label

    # Create labels if not already created
    if not image_label:
        image_label = tk.Label(root)
        image_label.pack(side=tk.LEFT, padx=10)

    if not mask_label:
        mask_label = tk.Label(root)
        mask_label.pack(side=tk.LEFT, padx=10)

    if not segmented_label:
        segmented_label = tk.Label(root)
        segmented_label.pack(side=tk.LEFT, padx=10)

    # Display the original image
    img_index = random.randint(0, len(images) - 1)
    original_img = Image.fromarray(np.squeeze(images[img_index], axis=2) * 255)
    # Convert to grayscale for display
    original_img = original_img.convert("L")
    original_img_tk = ImageTk.PhotoImage(original_img)
    image_label.config(image=original_img_tk)
    image_label.image = original_img_tk

    # Display the mask image
    mask_img = Image.fromarray(
        np.squeeze(masks[img_index], axis=2) * 255)
    mask_img = mask_img.convert("L")
    mask_img_tk = ImageTk.PhotoImage(mask_img)
    mask_label.config(image=mask_img_tk)
    mask_label.image = mask_img_tk

    # Create an RGBA image with a transparent background
    segmented_img = Image.new("RGBA", original_img.size)

    # Paste the original image onto the RGBA image
    segmented_img.paste(original_img, (0, 0))

    # Highlight damaged area with orange color
    # RGBA values for orange with semi-transparency
    orange_color = (255, 165, 0, 128)
    damaged_area = Image.fromarray(
        (np.squeeze(masks[img_index], axis=2) * 255).astype(np.uint8))
    damaged_area = damaged_area.convert("L")
    segmented_img.paste(
        orange_color, (0, 0, original_img.width, original_img.height), damaged_area)

    segmented_img_tk = ImageTk.PhotoImage(segmented_img)
    segmented_label.config(image=segmented_img_tk)
    segmented_label.image = segmented_img_tk


root = tk.Tk()
root.title("Model Teaching UI")

# Display the original image when UI starts
model_info_label = tk.Label(root, text=get_model_info())
model_info_label.pack(pady=10)

teach_button = tk.Button(root, text="Teach Model",
                         command=teach_model_thread)
teach_button.pack(pady=20)

show_graph_button = tk.Button(
    root, text="Show Graph", command=show_graph_button_click)
show_graph_button.pack(pady=20)

show_image_button = tk.Button(
    root, text="Show Image", command=show_image_button_click)
show_image_button.pack(pady=20)


root.mainloop()
