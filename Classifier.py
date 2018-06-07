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

model = joblib.load('new_KMeans.pkl')
model.set_params(verbose=False)
k = 500

for path in os.listdir(data_folder):
    if not path.startswith('.'):
        print('-------------------')
        print(path)
        print('-------------------')

        for feat in os.listdir(os.path.join(data_folder, path)):
            if feat.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_folder, path, feat))
                labels = model.predict(df.values)
                hist, _ = np.histogram(labels, k, density=True)
                X.append(hist)
                y.append(data_labels[path])

clf = svm.LinearSVC(verbose=True)

clf.fit(X, y)

joblib.dump(clf, 'new_LinearSVC.pkl')