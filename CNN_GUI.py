import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf

# Load CIFAR-10 data and model
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
model = tf.keras.models.load_model('cifar10_model.h5')  # Make sure the model is saved at this path

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_image(index, dataset):
    """ Load the image and label at the given index from the dataset. """
    image = Image.fromarray(dataset[index])
    label = class_names[train_labels[index][0]]
    return image, label


def resize_and_predict(image):
    """ Resize the custom image and predict using the CIFAR-10 model. """
    image = image.resize((640, 640), Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array[np.newaxis, ...]  # Add batch dimension

    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


def update_image(index):
    """ Update the image and label on the GUI. """
    image, label = load_image(index, train_images)
    image = image.resize((320, 320), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # keep a reference so it's not garbage collected
    label_text.set(f'Label: {label} (Image {index + 1} of {len(train_images)})')


def navigate_image(step):
    """ Navigate through the dataset. """
    global current_index
    current_index = (current_index + step) % len(train_images)
    update_image(current_index)


def open_image():
    """ Open a file dialog to select a custom image and display its prediction. """
    filepath = filedialog.askopenfilename()
    if filepath:
        image = Image.open(filepath)
        display_image = image.resize((320, 320), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        image_label.config(image=photo)
        image_label.image = photo  # keep a reference so it's not garbage collected

        predicted_class = resize_and_predict(image)
        label_text.set(f'Predicted Label: {predicted_class}')


# Initialize Tkinter window
root = tk.Tk()
root.title("CIFAR-10 Image Viewer")

# Initialize current index
current_index = 0

# Create a label for image and label text
image_label = tk.Label(root)
image_label.pack()

label_text = tk.StringVar()
tk.Label(root, textvariable=label_text).pack()

# Button to load custom image
load_button = tk.Button(root, text="Load Image", command=open_image)
load_button.pack(side=tk.BOTTOM, padx=5)
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

prev_button = tk.Button(button_frame, text="Previous", command=lambda: navigate_image(-1))
prev_button.pack(side=tk.LEFT, padx=5)

next_button = tk.Button(button_frame, text="Next", command=lambda: navigate_image(1))
next_button.pack(side=tk.LEFT, padx=5)

# Initial update of the image
update_image(current_index)

# Start the GUI loop
root.mainloop()
