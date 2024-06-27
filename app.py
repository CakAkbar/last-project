import streamlit as st
import pandas as pd
import joblib

# Memuat model dari file pickle
# model = joblib.load('naive_bayes_model.pkl')
model = pd.read_csv("data-clean.csv")

# Membuat antarmuka pengguna
st.title('Prediksi Kanker Payudara')

age = st.selectbox('Umur:', ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])
menopause = st.selectbox('Menopause:', ['lt40', 'ge40', 'premeno'])
tumor_size = st.selectbox('Ukuran Tumor:', ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'])
inv_nodes = st.selectbox('Kelenjar Getah Bening:', ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'])
node_caps = st.selectbox('Kapsul Node:', ['yes', 'no'])
deg_malig = st.selectbox('Tingkat Keganasan:', ['1', '2', '3'])
breast = st.selectbox('Letak Tumor Payudara:', ['left', 'right'])
breast_quad = st.selectbox('Letak Quadran Payudara:', ['left-up', 'left-low', 'right-up', 'right-low', 'central'])
irradiat = st.selectbox('Pengobatan Radiasi:', ['yes', 'no'])

if st.button('Prediksi'):
    # Mengambil data dari form
    data = {
        'age': age,
        'menopause': menopause,
        'tumor-size': tumor_size,
        'inv-nodes': inv_nodes,
        'node-caps': node_caps,
        'deg-malig': deg_malig,
        'breast': breast,
        'breast-quad': breast_quad,
        'irradiat': irradiat
    }
    
    # Membuat dataframe untuk prediksi
    df = pd.DataFrame([data])
    
    # Melakukan prediksi
    prediction = model.predict(df)[0]
    
    # Menampilkan hasil prediksi
    if prediction == 'no-recurrence-events':
        st.success('Tidak ada kejadian kekambuhan.')
    elif prediction == 'recurrence-events':
        st.warning('Ada kejadian kekambuhan.')
    else:
        st.info('Hasil tidak diketahui.')

# Jalankan dengan perintah: streamlit run app.py
