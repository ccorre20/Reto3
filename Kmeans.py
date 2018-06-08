# Developed by Camilo Correa Restrepo
# ccorre20@eafit.edu.co
# Version 2

# Kmeans.py
# This simple script trains a Kmeans model on a collection of csv that should contain, row by row, the vectors
# that correspond to the sift features extracted from an image. This will then be used to find the "k" most
# representative features, which will then be used to categorize every feature extracted from a certain image.
# This in turn will allow the creation of histogram where the relative occurence of each of these features will be
# represented by the value of each bin in the histogram.

import dask.dataframe as dd
import joblib
from sklearn.cluster import MiniBatchKMeans

data_path = 'data'

# Number of clusters, thus representing the number of visual words that are of interest.
k = 500

model_name = 'new_KMeans_'+str(k)+'.pkl'

print(model_name)

# Use dask to lazily load all the data at once, and thus avoid overloading the memory.
df = dd.read_csv(data_path + '/*/*.csv')

# Create and train the model, using dasks lazy loading of data.
model = MiniBatchKMeans(n_clusters=k, verbose=True)
model.fit(df.values)

# Export the trained model, so it can be used in a later stage.
print("exporting")
joblib.dump(model, model_name)
print("done")


