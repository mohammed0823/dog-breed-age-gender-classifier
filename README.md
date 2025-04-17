# dog-breed-age-gender-classifier

This repository contains a machine learning pipeline aimed at classifying dog images by their age, breed, and gender. The project uses Convolutional Neural Networks (CNNs) and various preprocessing techniques to enhance model performance. The dataset, consisting of labeled dog images, was cleaned and augmented for better training results.

## Files

### 1. **VGGNet-19 Implementation.py**
   - Implements the VGGNet-19 model with TensorFlow/Keras.
   - Prepares data pipeline, processes images, and trains the model for classification of dog images by breed, age, and gender.
   - Utilizes transfer learning and adds custom output layers for multi-label classification.

### 2. **File Cleaning.py**
   - A script for cleaning and renaming image files in the dataset.
   - Normalizes breed names and formats image labels to be consistent.
   - Deletes breeds with fewer than five images to maintain dataset quality.

### 3. **Dog Image Classification Report.pdf**
   - A detailed report that covers the entire project, from data exploration to model implementation and results.
   - Includes challenges faced during the project and future improvements.

### 4. **PreprocessingTestPillow.py**
   - Handles image preprocessing tasks like resizing, grayscale conversion, edge enhancement, and noise addition.
   - Includes background removal algorithm based on contrast and image masking.
   - Performs image augmentation techniques like flipping, rotating, and adding noise to expand the dataset.

## Setup

### Requirements
- Python 3.x
- TensorFlow
- Pandas
- scikit-learn
- NumPy
- Pillow
- KaggleHub

## Running the Scripts

### 1. **File Cleaning:**
   - The File Cleaning.py script will clean and rename the image files based on a standardized naming convention. Ensure that the dataset path is provided.

### 2. **Preprocessing:**
   - The PreprocessingTestPillow.py script can be used for image preprocessing and augmentation. It includes a custom background removal method to enhance the model's performance.

### 3. **VGGNet-19 Implementation:**
   - Run the VGGNet-19 Implementation.py to train the model. It will ask for the path to the images and will automatically preprocess and train the model.

## Conclusion

This project successfully implements a model to classify dog images by their breed, age, and gender. The VGGNet-19 architecture was chosen for its robust feature extraction capabilities, and preprocessing techniques like background blurring and data augmentation were used to enhance model performance.

## Future Improvements
   - Improve breed classification accuracy by balancing the dataset.
   - Experiment with different CNN architectures and classifiers, like SVM, for better accuracy.
   - Implement more advanced preprocessing techniques like image segmentation for feature enhancement.
