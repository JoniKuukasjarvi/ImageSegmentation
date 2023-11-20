import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import filedialog
from threading import Thread
from contextlib import redirect_stdout
from io import StringIO
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image, ImageTk, ImageColor
from keras.utils import plot_model
from PIL import Image, ImageTk
import random

image_label = None
mask_label = None
segmented_label = None
teach_button = None
root = None
history = None
show_summary_button = None
text_widget = None


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
    x = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(inputs)
    c1_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    c2_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    c3_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    c4_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)

    # Expansive Path
    x = keras.layers.Conv2DTranspose(
        512, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c4_residue])
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2DTranspose(
        256, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c3_residue])
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c2_residue])
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c1_residue])
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_shape, num_classes=1)

# Model Config
epochs = 2

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "teras1.weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    keras.callbacks.EarlyStopping(patience=30, verbose=1),
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

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
    )
    print("\nTraining finished.")
    teach_button.config(state=tk.NORMAL)


model = make_model(input_shape=image_shape, num_classes=1)


def get_model_info():
    model_name = callbacks[0].filepath if callbacks and len(
        callbacks) > 0 else "undefined"
    return f"Current Model: {model_name}"


def teach_model_thread():
    training_thread = Thread(target=train_model)
    training_thread.start()


def show_graph_button_click():
    if history is not None:
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


def show_image_button_click():  # For random image
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

    # Ensure the image_label is updated
    root.update_idletasks()


def get_model_summary():
    with StringIO() as buffer, redirect_stdout(buffer):
        model.summary()
        return buffer.getvalue()


def toggle_summary():
    global text_widget

    if text_widget:
        text_widget.pack_forget()  # Remove the text widget from the UI
        text_widget = None
    else:
        # Display model summary at the beginning of the text widget
        model_summary = get_model_summary()
        text_widget = tk.Text(root, wrap=tk.WORD, width=60, height=20)
        text_widget.insert(tk.END, model_summary)
        text_widget.pack()


def save_model_img():
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[
                                                 ("PNG files", "*.png")],
                                             title="Save Model Architecture Plot")
    if file_path:
        if not file_path.endswith('.png'):
            file_path += '.png'
        plot_model(model, to_file=file_path,
                   show_shapes=True, show_layer_names=True)
        print(f"Model visualization saved at {file_path}")


def create_predicted_mask(image_path, model):
    global root, image_label, mask_label

    # Ensure image_label is created
    if not image_label:
        image_label = tk.Label(root)
        image_label.pack(side=tk.LEFT, padx=10)

    # Ensure mask_label is created
    if not mask_label:
        mask_label = tk.Label(root)
        mask_label.pack(side=tk.LEFT, padx=10)

    # Load the user-inputted image
    user_image = Image.open(image_path)
    user_image = user_image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Preprocess the image for model prediction
    img_array = image.img_to_array(user_image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Display the original image
    original_img_tk = ImageTk.PhotoImage(user_image)
    image_label.config(image=original_img_tk)
    image_label.image = original_img_tk
    root.update_idletasks()  # Ensure the image_label is updated

    # Generate predictions using the model
    predicted_mask = model.predict(img_array)[0]

    threshold = 0.5
    predicted_mask[predicted_mask >= threshold] = 1
    predicted_mask[predicted_mask < threshold] = 0

    # Display the predicted mask
    predicted_mask_img = Image.fromarray(
        (predicted_mask.squeeze() * 255).astype(np.uint8))
    predicted_mask_img = predicted_mask_img.convert("L")
    predicted_mask_img_tk = ImageTk.PhotoImage(predicted_mask_img)
    mask_label.config(image=predicted_mask_img_tk)
    mask_label.image = predicted_mask_img_tk


def upload_and_predict():
    global root, image_label, mask_label

    if image_label:
        image_label.pack_forget()
        image_label = None

    if mask_label:
        mask_label.pack_forget()
        mask_label = None

    image_path = filedialog.askopenfilename(
        title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if image_path:
        create_predicted_mask(image_path, model)


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

show_summary_button = tk.Button(root, text="Summary", command=toggle_summary)
show_summary_button.pack(pady=20)

save_model_button = tk.Button(
    root, text="Create Model Image", command=save_model_img)
save_model_button.pack(pady=20)

upload_image_button = tk.Button(
    root, text="Upload Image", command=upload_and_predict)
upload_image_button.pack(pady=20)


root.mainloop()
