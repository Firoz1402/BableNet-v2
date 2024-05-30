import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import sys
from sklearn.utils import shuffle


def load_and_preprocess_images(dataset_folder, image_size):
    image_data = []
    labels = []

    for data_type in ["images_of_mammals"]:
        data_folder = os.path.join(dataset_folder, data_type)

        for label_idx, label_name in enumerate(os.listdir(data_folder)):
            label_folder = os.path.join(data_folder, label_name)

            for image_filename in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                
                # Check if the image was loaded successfully
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                
                # Resize the image to the desired size (128x128 pixels)
                image = cv2.resize(image, image_size)
                image = image.flatten()  # Flatten the image
                image_data.append(image)
                labels.append(label_name)

    return np.array(image_data), np.array(labels)




def convert_image_to_csv(dataset_folder,desired_image_size):
    # Load and preprocess images
    X, y = load_and_preprocess_images(dataset_folder,desired_image_size)

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Create a DataFrame for the pixel values and labels
    data = pd.DataFrame(X)
    data.insert(0, "label", y)

    # Save the DataFrame to a CSV file
    csv_filename = "dataset.csv"
    data.to_csv(csv_filename, index=False)

    print(f"Dataset saved to {csv_filename}.")



def get_dataset_folder():
    while True:
        dataset_folder = input("Enter the dataset folder path: ")
        if os.path.isdir(dataset_folder):
            print(f"Dataset folder: {dataset_folder}")
            return dataset_folder
        else:
            print("Invalid folder path. Please enter a valid path.")

def get_desired_image_size():
    while True:
        try:
            input_string = input("Enter the desired image size (width, height) separated by a comma: ")
            width, height = map(int, input_string.split(','))
            desired_image_size = (width, height)
            print(f"Desired image size: {desired_image_size}")
            return desired_image_size
        except ValueError:
            print("Invalid input. Please enter two integers separated by a comma.")


def main():
    desired_image_size = get_desired_image_size()
    dataset_folder = get_dataset_folder()
    convert_image_to_csv(dataset_folder, desired_image_size)


if __name__ == "__main__":
    main()
