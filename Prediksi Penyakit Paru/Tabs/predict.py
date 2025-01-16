import streamlit as st
from web_functions import predict

def app(df, x, y):
    st.title("Prediksi Terkena Penyakit Paru-paru")
    st.write("Silahkan masukkan input Prediksi Terkena Penyakit Paru-paru dengan benar!!!")
        
    # Input Kolom
    Usia = st.number_input('Input Nilai Usia (Tua = 1, Muda = 0)', min_value=0, max_value=1)
    Jenis_Kelamin = st.number_input('Input Nilai Jenis_Kelamin (Pria = 0, Wanita = 1)', min_value=0, max_value=1)
    Merokok = st.number_input('Input Nilai Merokok (Pasif = 1, Aktif = 0)', min_value=0, max_value=1)
    Bekerja = st.number_input('Input Nilai Bekerja (Tidak = 0, Ya = 1)', min_value=0, max_value=1)
    Rumah_Tangga = st.number_input('Input Nilai Rumah_Tangga (Ya = 1, Tidak = 0)', min_value=0, max_value=1)
    Aktivitas_Begadang = st.number_input('Input Nilai Aktivitas_Begadang (Ya = 1, Tidak=0)', min_value=0, max_value=1)
    Aktivitas_Olahraga = st.number_input('Input Nilai Aktivitas_Olahraga (Sering = 1, Jarang = 0)', min_value=0, max_value=1)
    Asuransi = st.number_input('Input Nilai Asuransi (Ada = 0, Tidak = 1)', min_value=0, max_value=1)
    Penyakit_Bawaan = st.number_input('Input Nilai Penyakit_Bawaan (Tidak = 1, Ada = 0)', min_value=0, max_value=1)

    features = (Usia, Jenis_Kelamin, Merokok, Bekerja, Rumah_Tangga, Aktivitas_Begadang, Aktivitas_Olahraga, Asuransi, Penyakit_Bawaan)

    # Tombol Prediksi
    if st.button("Prediksi"):
            prediction, score = predict(x, y, features)
            st.info("Prediksi Sukses...")

            if prediction == 1:
                st.warning("Orang tersebut rentan terkena penyakit paru-paru")
            else:
                st.success("Orang tersebut relatif aman dari penyakit paru-paru")

            st.write("Model yang digunakan memiliki tingkat akurasi", (score * 100), "%")

# Contoh Penggunaan:
# app(df, x, y)  # Pastikan untuk mengganti df, x, dan y dengan variabel yang sesuai dari proyek Anda.
