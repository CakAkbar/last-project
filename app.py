import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

age = st.selectbox('Umur:', ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])
menopause = st.selectbox('Menopause:', ['lt40', 'ge40', 'premeno'])
tumor_size = st.selectbox('Ukuran Tumor:', ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'])
inv_nodes = st.selectbox('Kelenjar Getah Bening:', ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'])
node_caps = st.selectbox('Kapsul Node:', ['yes', 'no'])
deg_malig = st.selectbox('Tingkat Keganasan:', ['1', '2', '3'])
breast = st.selectbox('Letak Tumor Payudara:', ['left', 'right'])
breast_quad = st.selectbox('Letak Quadran Payudara:', ['left-up', 'left-low', 'right-up', 'right-low', 'central'])
irradiat = st.selectbox('Pengobatan Radiasi:', ['yes', 'no'])


# Misalkan df_breast_cancer adalah DataFrame yang sudah ada
df_breast_cancer = pd.read_csv('data-clean.csv')

# Menambahkan kolom 'class' berdasarkan kondisi 'deg-malig'
df_breast_cancer['class'] = df_breast_cancer.apply(lambda row: 'no-recurrence-events' if row['deg-malig'] < 2 else 'recurrence-events', axis=1)

# Kolom target
target_column = 'class'

# Memisahkan fitur (X) dan label (y)
X = df_breast_cancer.drop(target_column, axis=1)
y = df_breast_cancer[target_column]

# Membagi dataset menjadi data latih dan data uji (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengidentifikasi kolom kategorikal dan biner
categorical_features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'breast-quad']
binary_features = ['node-caps', 'breast', 'irradiat']

# Membuat transformer untuk preprocessing
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
binary_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Menggabungkan transformer dalam sebuah kolom transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ],
    remainder='passthrough'
)
# Pipeline untuk Naive Bayes
pipeline_nb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ('classifier', GaussianNB())
])

# Melatih model Naive Bayes dengan data latih
pipeline_nb.fit(X_train, y_train)

# Mengevaluasi model Naive Bayes dengan data uji
y_pred_nb = pipeline_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Akurasi model Naive Bayes: {accuracy_nb:.2f}")