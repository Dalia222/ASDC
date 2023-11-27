import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

data_dir = "D:/ASDC_internship/Second_task/"
image_dir = os.path.join(data_dir, "jpg")
label_path = os.path.join(data_dir, "imagelabels.mat")
setid_path = os.path.join(data_dir, "setid.mat")

labels = loadmat(label_path)['labels'][0]

class_indices = {class_id: idx + 1 for idx, class_id in enumerate(sorted(np.unique(labels)))}

images = []
category_labels = []

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)

    # Skip images that don't follow the expected naming convention
    match = re.match(r'image_(\d+)\.jpg', img_name)
    if not match:
        print(f"Skipping {img_name} due to unexpected naming.")
        continue

    img_label = match.group(1)

    category_labels.append(class_indices.get(int(img_label), 0))
    images.append(img_path)  # Add the image path to the list

assert len(images) == len(category_labels), "Number of images and labels do not match."

print("Number of images:", len(images))
print("Number of labels:", len(category_labels))

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, category_labels, test_size=0.1, random_state=42
)

df_train = pd.DataFrame({'filename': train_images, 'label': train_labels})
df_val = pd.DataFrame({'filename': val_images, 'label': val_labels})

df_train['label'] = df_train['label'].astype(str)
df_val['label'] = df_val['label'].astype(str)

# Rest of your code...


# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

img_size = (224, 224)
batch_size = 16

train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=None,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df_val,
    directory=None,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Convolutional neural network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(class_indices), activation='softmax'))

model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=10
)

print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])

print("Training Loss:", history.history['loss'][-1])
print("Validation Loss:", history.history['val_loss'][-1])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
