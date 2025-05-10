import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops

base_path = './output'

# Parametry
distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # kąty w radianach: 0°, 45°, 90°, 135°
properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

features_list = []

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(category_path, filename)

            image = io.imread(filepath)
            gray = color.rgb2gray(image) if image.ndim == 3 else image

            # Redukcja jasności do 5 bitów (64 poziomy)
            gray_5bit = (gray * 63).astype(np.uint8)

            # Obliczanie macierzy GLCM
            glcm = graycomatrix(gray_5bit, distances=distances, angles=angles,
                                levels=64, symmetric=True, normed=True)

            # Obliczanie cech
            feature_vector = []
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                feature_vector.extend(prop_values.flatten())

            feature_vector.append(category)

            features_list.append(feature_vector)

columns = []
for prop in properties:
    for distance in distances:
        for angle_deg in [0, 45, 90, 135]:
            columns.append(f'{prop}_d{distance}_a{angle_deg}')
columns.append('category')

features_df = pd.DataFrame(features_list, columns=columns)

# Zapis do pliku CSV
features_df.to_csv('texture_features.csv', index=False)

print("Wyniki zapisane do 'texture_features.csv'.")