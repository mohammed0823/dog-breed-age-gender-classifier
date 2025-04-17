import os
import re
import numpy as np
from collections import defaultdict
import pandas as pd  # Added for DataFrame operations

def cleanFilename(folder_path):
    # Dictionary to track breed counts and associated file paths
    breed_counts = defaultdict(list)

    # DataFrame for breed adjustments
    df = pd.DataFrame(columns=["breed"])

    # Step 1: Process and clean filenames
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Full path to the image
                file_path = os.path.join(root, filename)

                # Split the filename into base name and extension
                base_name, file_extension = os.path.splitext(filename)

                # Make sure the extension is in lowercase to handle case consistency
                file_extension = file_extension.lower()

                # Start cleaning the filename
                clean_filename = base_name.replace(" copy ", "")
                clean_filename = clean_filename.replace(" copy", "")
                clean_filename = clean_filename.upper()

                # Fixing gender
                clean_filename = re.sub(r'\s*\(\d+\)', '', clean_filename)  # Remove (number)
                clean_filename = clean_filename.replace("MALE", "M").replace("M1", "M")
                clean_filename = clean_filename.replace("F1", "F").replace("F2", "F")
                clean_filename = clean_filename.replace("M(10_", "M").replace("M91)", "M")
                clean_filename = clean_filename.replace("M (3_", "M")

                # Age modifications
                clean_filename = clean_filename.replace("1.25 ", "1-2 ").replace("1.5 ", "1-2 ")
                clean_filename = clean_filename.replace("0.83 ", "0-1 ").replace("0.2 ", "0-1 ")
                clean_filename = clean_filename.replace(" 2.5 ", " 2-5 ").replace(" 0.5 ", " 0-1 ")
                clean_filename = clean_filename.replace(" 1 ", " 0-1 ").replace(" 0.5-1 ", " 0-1 ")

                for i in np.arange(0, 1, 0.01):
                    clean_filename = clean_filename.replace(f"{i:.2f} ", "0-1 ")

                for i in range(1, 5):
                    clean_filename = clean_filename.replace(f" {i} ", " 2-5 ")

                for i in range(4, 8):
                    clean_filename = clean_filename.replace(f" {i} ", " 5-7 ")

                for i in range(7, 20):
                    clean_filename = clean_filename.replace(f" {i} ", " 8+ ")

                # Breed modifications
                clean_filename = clean_filename.replace("CROSS ", "CROSSBREED ")
                clean_filename = clean_filename.replace("CHIHUAHUA SHORT HAIR ", "CHIHUAHUA ")
                clean_filename = clean_filename.replace("CHIHUAHUA LONG HAIR ", "CHIHUAHUA ")
                clean_filename = clean_filename.replace("LAB ", "LABRADOR ")
                clean_filename = clean_filename.replace("JACK RUSSEL TERRIER ", "JACK RUSSELL TERRIER ")

                # Split the cleaned filename by spaces, extracting breed, age, and gender
                parts = clean_filename.rsplit(' ', 2)

                if len(parts) == 3:
                    breed = parts[0]  # Breed name
                    age = parts[1]
                    gender = parts[2].split('.')[0]

                    # Add breed to DataFrame for normalization
                    new_row = pd.DataFrame({"breed": [breed]})
                    df = pd.concat([df, new_row], ignore_index=True)

                    # Track breed counts and file paths
                    breed_counts[breed].append(file_path)

                    # Handle duplicates by appending a counter to the file name
                    counter = 1
                    new_file_name = f"{breed} {age} {gender}{file_extension}"
                    new_file_path = os.path.join(root, new_file_name)

                    while os.path.exists(new_file_path):
                        new_file_name = f"{breed} {age} {gender}_{counter}{file_extension}"
                        new_file_path = os.path.join(root, new_file_name)
                        counter += 1

                    print(f"Renaming: {file_path} -> {new_file_path}")
                    os.rename(file_path, new_file_path)
                else:
                    print(f"Filename '{filename}' does not match the expected format 'breed age gender'")

    # Normalize breed names using DataFrame logic
    df.loc[df["breed"] == "JACK RUSSELL", "breed"] = "JACK RUSSELL TERRIER"
    df.loc[df["breed"] == "SMOOTH COLLIE", "breed"] = "COLLIE"
    df.loc[df["breed"] == "AMERICAN BULLDOG", "breed"] = "BULLDOG"
    df.loc[df["breed"] == "CROSSBREED BREED", "breed"] = "CROSSBREED"
    df.loc[df["breed"] == "AMERICAN BULL DOG", "breed"] = "BULLDOG"
    df.loc[df["breed"] == "JACK RUSSERL TERRIER", "breed"] = "JACK RUSSELL TERRIER"

    # Step 2: Delete images for breeds with 5 or fewer files
    for breed, files in breed_counts.items():
        if len(files) <= 5:
            print(f"Deleting all images for breed '{breed}' (only {len(files)} images)")
            for file_path in files:
                if os.path.exists(file_path):  # Check if the file exists before attempting to delete
                    os.remove(file_path)
                else:
                    print(f"File not found: {file_path}")

    print("Processing completed successfully!")

# Example usage
Filename = input("Please enter the base directory for the images: ")
Filename = Filename.replace('"', '').replace("'", '')
cleanFilename(Filename)