# Developed by Camilo Correa Restrepo
# ccorre20@eafit.edu.co
# Version 2

# SIFT_Extract.py
# This scrip takes a collection of images, categorized and separated in folder accordingly, and extracts their SIFT
# features, creating a .csv file with them. This will then be used to create and train both of the subsequent models
# of this solution.

import pandas as pd
import os
import cv2

image_path = 'distilled_images'

data_folder = 'data'

sift = cv2.xfeatures2d.SIFT_create()

# This for loop, in essence enters every folder in the image_path
for folder in os.listdir(image_path):
    if not folder.startswith('.'):
        print('-------------------')
        print(folder)
        print('-------------------')

        # This for loop goes over every image within each of the folders extracted by the previous loop.
        for image in os.listdir(os.path.join(image_path, folder)):
            # This ignores any file that is hidden or is not an image.
            if not image.startswith('.') and not image.endswith('.csv'):
                # Read the image
                img = cv2.imread(os.path.join(image_path, folder, image))
                # Convert it to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Extract keypoints (but disregard them, hence _) and descriptors of said keypoints as a 2d array.
                _, des = sift.detectAndCompute(gray, None)

                # Turn that array into a pandas dataframe and export it a as a csv
                df = pd.DataFrame(des)
                print(os.path.join(data_folder, folder, str(image.split('.')[0]) + '.csv'))
                df.to_csv(os.path.join(data_folder, folder, str(image.split('.')[0]) + '.csv'), header=False,
                          index=False)

print('All done')
