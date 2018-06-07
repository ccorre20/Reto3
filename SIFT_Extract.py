import pandas as pd
import os
import cv2

image_path = 'distilled_images'

data_folder = 'data'

sift = cv2.xfeatures2d.SIFT_create()

for folder in os.listdir(image_path):
    if not folder.startswith('.'):
        print('-------------------')
        print(folder)
        print('-------------------')

        for image in os.listdir(os.path.join(image_path, folder)):
            if not image.startswith('.') and not image.endswith('.csv'):
                img = cv2.imread(os.path.join(image_path, folder, image))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, des = sift.detectAndCompute(gray, None)
                df = pd.DataFrame(des)
                print(os.path.join(data_folder, folder, str(image.split('.')[0]) + '.csv'))
                df.to_csv(os.path.join(data_folder, folder, str(image.split('.')[0]) + '.csv'), header=False,
                          index=False)

print('All done')
