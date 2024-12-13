import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization)

#####################
# Configuration
#####################
train_dir = 'dataset/train'
test_dir = 'dataset/test'

target_size = (48, 48)     
color_mode = 'grayscale'   
batch_size = 64
num_epochs = 50

# Check if train and test directories exist
if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
    raise FileNotFoundError("Ensure that 'dataset/train' and 'dataset/test' directories exist!")

#####################
# Data Generators
#####################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode=color_mode,
    class_mode='categorical',
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    color_mode=color_mode,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

num_classes = train_generator.num_classes
print("Number of emotion classes:", num_classes)
print("Class indices:", train_generator.class_indices)

#####################
# Model Definition
#####################
model = Sequential()

# Block 1
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(target_size[0], target_size[1], 1 if color_mode=='grayscale' else 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

#####################
# Compilation
#####################
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#####################
# Training
#####################
# Without a validation set, we just train on the training data for the specified number of epochs.
history = model.fit(
    train_generator,
    epochs=num_epochs
)

# Save the model after training
model.save('emotion_recognition_model_final.h5')

#####################
# Evaluation
#####################
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)
