from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns
    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)

    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test = preprocessor.transform(X_test)
    # Simpan pipeline
    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test


from sklearn.datasets import load_iris
import pandas as pd
# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()

# Contoh Penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

from joblib import dump, load

def inference(new_data, load_path):
    # Memuat pipeline preprocessing
    preprocessor = load(load_path)
    print(f"Pipeline preprocessing dimuat dari: {load_path}")

    # Transformasi data baru
    transformed_data = preprocessor.transform(new_data)
    return transformed_data

def inverse_transform_data(transformed_data, load_path, new_data_columns):
    preprocessor = load(load_path)
    # Membalik transformasi hanya untuk fitur numerik
    numeric_transformer = preprocessor.named_transformers_['num']['scaler']
    numeric_columns = new_data_columns[:len(numeric_transformer.mean_)]  # Asumsi fitur numerik ada di awal
    transformed_numeric_data = transformed_data[:, :len(numeric_columns)]

    # Inverse transform untuk data numerik
    original_numeric_data = numeric_transformer.inverse_transform(transformed_numeric_data)

    # Gabungkan kembali data numerik dengan data kategoris (jika ada)
    inversed_data = pd.DataFrame(original_numeric_data, columns=numeric_columns)
    return inversed_data

import numpy as np
# Jalankan preprocessing
pipeline_path = 'preprocessor_pipeline.joblib'
col = pd.read_csv('data.csv')
# Daftar data
new_data = [5.1, 3.5, 1.4, 0.2]

# Mengubah menjadi numpy.ndarray
new_data = np.array(new_data)

new_data = pd.DataFrame([new_data], columns=col.columns)
# Lakukan inference
transformed_data = inference(new_data, pipeline_path)

# Inverse transform data
inversed_data = inverse_transform_data(transformed_data, pipeline_path, new_data.columns)

# Output hasil preprocessing dan inference
print("Data setelah preprocessing (training):")
print(new_data)
print("\nData baru setelah transformasi:")
print(transformed_data)
print("\nData setelah inverse transform:")
print(inversed_data)