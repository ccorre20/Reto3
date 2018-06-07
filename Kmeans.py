import dask.dataframe as dd
import os
import joblib
from sklearn.cluster import MiniBatchKMeans

data_path = 'data'

df = dd.read_csv(data_path + '/*/*.csv')

model = MiniBatchKMeans(n_clusters=500, verbose=True)

model.fit(df.values)

print("exporting")
joblib.dump(model, "new_KMeans.pkl")
print("done")


