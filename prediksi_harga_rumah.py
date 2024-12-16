"""
Prediksi_Harga_Rumah.ipynb
Created by "Ahmad Ja'far Ali"

"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""## **Load DATASET**"""

dataFrame = pd.read_csv('data_rumah.csv')
print("Berisi 546 data yang terdiri dari:")
print(dataFrame.head())

"""# **Preprocessing**"""

print("\nMemeriksa nilai data yang hilang:")
print(dataFrame.isnull().sum())

"""# MENGISI DATA YANG HILANG"""

dataFrame.fillna(dataFrame.median(), inplace=True)
df = pd.get_dummies(dataFrame, drop_first=True)

"""# Memisah fitur (X) dan target (Y)"""

X = dataFrame[['luas', 'kasur','km']]
y = dataFrame['harga']

"""# Memisah ***Data Pelatihan*** & ***Data Uji***"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# ***Melatih Model*** & ***Evaluasi Model***"""

print("\nMelatih model Random Forest Regressor")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluasi Model:")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

"""## Menyimpan Model menggunakan ***pickle*** dan Memuat Model kemudian menguji Prediksi"""

with open('Random_Forest_Regressor_Model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nModel disimpan dengan format 'Random_Forest_Regressor_Model.pkl'")

with open('Random_Forest_Regressor_Model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
print("\nModel berhasil dimuat")

"""# Memprediksi dengan input data baru"""

new_data = np.array([[1500, 3, 5]])
prediksi_harga = loaded_model.predict(new_data)
print(f"\nHarga yang Diprediksi untuk data baru: {prediksi_harga}")