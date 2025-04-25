from sklearn.datasets import load_iris
import pandas as pd
from preprocessing import preprocess_data
from joblib import dump, load
import numpy as np
 
# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()

# Contoh Penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

def inference(new_data, load_path):
    # Memuat pipeline preprocessing
    preprocessor = load(load_path)
    print(f"Pipeline preprocessing dimuat dari: {load_path}")
 
    # Transformasi data baru
    transformed_data = preprocessor.transform(new_data)
    return transformed_data

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