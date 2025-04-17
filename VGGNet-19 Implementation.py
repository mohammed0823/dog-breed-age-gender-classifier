import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split

# Set the base directory for the images (replace with your actual path)
base_dir = input("Please enter the base directory for the images: ")
base_dir = base_dir.replace('"', '').replace("'", '')

# List to hold extracted image data
image_data = []

# Define a function to extract labels from filenames
def extract_labels(folder_path):
    global image_data
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Full path to the image
            file_path = os.path.join(folder_path, filename)

            # Assuming filenames are cleaned and formatted as "breed age gender"
            parts = filename.rsplit(' ', 2)

            if len(parts) == 3:
                breed = parts[0]  # Breed name
                age = parts[1]    # Age category
                gender = parts[2].split('.')[0]  # Gender

                # Append the data to the list
                image_data.append({
                    'file_path': file_path,
                    'breed': breed,
                    'age': age,
                    'gender': gender
                })
            else:
                print(f"Filename '{filename}' does not match the expected format 'breed age gender'")

# Parse data and extract labels
for category in ['Adult', 'Senior', 'Young']:
    for subfolder in ['ProcessedData', 'AugmentedData', 'BackgroundRemoved']:
        folder_path = os.path.join(base_dir, category, subfolder)
        extract_labels(folder_path)

# Create label mappings
age_mapping = {label: idx for idx, label in enumerate(sorted(set(item['age'] for item in image_data)))}
breed_mapping = {label: idx for idx, label in enumerate(sorted(set(item['breed'] for item in image_data)))}
gender_mapping = {label: idx for idx, label in enumerate(sorted(set(item['gender'] for item in image_data)))}

# Efficient Data Pipeline using tf.data.Dataset
def process_image(file_path):
    # Load the image as a tensor
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Force grayscale loading
    image = tf.image.resize(image, (256, 256)) / 255.0  # Resize and normalize
    image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB (3 channels)
    return image

def load_data(image_data):
    # Load and preprocess all images
    image_paths = [item['file_path'] for item in image_data]
    images = tf.data.Dataset.from_tensor_slices(image_paths)
    images = images.map(lambda x: process_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prepare labels
    age_labels = [age_mapping[item['age']] for item in image_data]
    breed_labels = [breed_mapping[item['breed']] for item in image_data]
    gender_labels = [gender_mapping[item['gender']] for item in image_data]

    labels = {
        'age_output': age_labels,
        'breed_output': breed_labels,
        'gender_output': gender_labels
    }

    labels = tf.data.Dataset.from_tensor_slices(labels)

    # Combine images and labels
    dataset = tf.data.Dataset.zip((images, labels))

    # Shuffle, batch, and cache the dataset for better performance
    dataset = dataset.shuffle(buffer_size=len(image_data)) \
                     .batch(32) \
                     .prefetch(tf.data.experimental.AUTOTUNE) \
                     .cache()  # Cache the dataset to avoid reloading every epoch

    return dataset

# Load the training and validation data
train_dataset, val_dataset = train_test_split(image_data, test_size=0.2, random_state=42)

train_dataset = load_data(train_dataset)
val_dataset = load_data(val_dataset)

# Create the model (VGG19)
base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Add custom output layers for multi-label classification (age, breed, gender)
age_output = Dense(len(age_mapping), activation='softmax', name='age_output')(x)
breed_output = Dense(len(breed_mapping), activation='softmax', name='breed_output')(x)
gender_output = Dense(len(gender_mapping), activation='softmax', name='gender_output')(x)

# Complete model definition
model = Model(inputs=base_model.input, outputs=[age_output, breed_output, gender_output])

# Freeze the VGG19 base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(),
              loss={'age_output': 'sparse_categorical_crossentropy',
                    'breed_output': 'sparse_categorical_crossentropy',
                    'gender_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)