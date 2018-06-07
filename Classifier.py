# Developed by Camilo Correa Restrepo
# ccorre20@eafit.edu.co
# Version 2

# Classifier.py
# This program creates a classifier to be used to determine what building an image corresponds to.
# This is done by obtaining the SIFT features from the image, then, those images are transformed into
# histograms, where each bin represents the relative frequency of occurence of these visual features
# of interest. In particular, they are created by categorizing each of the SIFT features into one of
# k most representative, as determinded by a kmeans model fit on a large dataset of images.

import os
import pandas as pd
import numpy as np
import joblib
from sklearn import svm

data_folder = 'data'
data_labels = {'18': 0.0, '19': 1.0, '26': 2.0, '38': 3.0, 'admi': 4.0,
               'agora': 5.0, 'audi': 6.0, 'biblio': 7.0, 'dogger': 8.0, 'idiomas': 9.0}

X = []
y = []

# This loads the Kmeans model, so each feature that is loaded, can be classified.
model = joblib.load('new_KMeans.pkl')
model.set_params(verbose=False)
k = 500

# This goes through all of the folders with the data path, where all the .csv files are.
for path in os.listdir(data_folder):
    if not path.startswith('.'):
        print('-------------------')
        print(path)
        print('-------------------')

        # This now goes through each image's features, which are in .csv
        for feat in os.listdir(os.path.join(data_folder, path)):
            # This disregards anything that isn't a csv.
            if feat.endswith('.csv'):
                # Use pandas to read the csv
                df = pd.read_csv(os.path.join(data_folder, path, feat))
                # Run the dataframe through the model to get the labels.
                # labels contains the corresponding label (0<->(k-1)) for each feature, as an array.
                labels = model.predict(df.values)
                # This is array is then transformed into a normalized histogram,
                # represented as its probability density function.
                hist, _ = np.histogram(labels, k, density=True)
                # Add this histogram's values as a sample in our X array.
                X.append(hist)
                # Add the correspoding label for the location.
                y.append(data_labels[path])

# Create and train a classifier, utilizing the histograms as the feature vectors.
clf = svm.LinearSVC(verbose=True)
clf.fit(X, y)

# Export the completed model for use in the web application.
joblib.dump(clf, 'new_LinearSVC.pkl')