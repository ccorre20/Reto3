The project contained within this folder is made up of a series of scripts for constructing clustering and classification models
for use in a web application for detecting the correspondence between images and several locations.

The other part of this project can be found at: https://github.com/ccorre20/web

Within this folder you will find:
.gitignore: Removes all the data from the repository so it can be transferred to github.

SIFT_Extract.py: This will take a collection of images and construct a series of csv files with their corresponding SIFT feature descriptors vertically stacked.

KMeans.py: Constructs a model that classifies what cluster each of the extracted features belongs to.

Classifer.py: This constructs a classification model, designed to allow to detect what location each image corresponds to. It also constructs the histograms it uses to train the model.

location-recognition-software.pdf: This is perhaps the most important document, this describes how the entireity of the system functions.

Execution:
The pipeline for executing these scripts is as follows:

1. With a correctly distributed folder of images (separated by type/location/etc), extract their features with SIFT_Extract.py
2. Run Kmeans.py with your desired number of clusters.
3. Run Classifier.py once you have a Kmeans model, allowing you to classify what location each image corresponds to.

With that you will models named new_LinearSVC_<k>.pkl, new_KMeans_<k>.pkl, with <k> being the number of clusters selected. With this you can now build applications which make use of these models.