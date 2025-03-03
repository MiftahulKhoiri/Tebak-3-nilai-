import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from collections import deque
import os

# Variabel global untuk menyimpan dataset
dataset_inputs = deque(maxlen=1000)  # Menyimpan input terakhir (maksimal 1000 data)
dataset_outputs = deque(maxlen=1000)  # Menyimpan output terakhir (maksimal 1000 data)

def validasi_input(user_input):
    """
    Memvalidasi input pengguna. Input harus berupa 3 angka antara 1 dan 6.
    """
    try:
        numbers = list(map(int, user_input.split()))
        if len(numbers) != 3 or any(num < 1 or num > 6 for num in numbers):
            return False
        return True
    except ValueError:
        return False

def simpan_data(input_data):
    """
    Menyimpan input dan output ke dataset.
    """
    try:
        if len(dataset_inputs) > 0:
            dataset_outputs.append(input_data)  # Output adalah input berikutnya
        dataset_inputs.append(input_data)  # Input adalah data sebelumnya
    except Exception as e:
        print(f"Gagal menyimpan data: {e}")

def latih_model_ensemble(X, y):
    """
    Melatih model ensemble untuk setiap angka (angka pertama, angka kedua, angka ketiga).
    Setiap angka diprediksi secara terpisah menggunakan 3 model: Random Forest, Gradient Boosting, dan Linear Regression.
    """
    try:
        # Inisialisasi model untuk setiap angka
        model_rf_1 = RandomForestRegressor(n_estimators=100, random_state=42)
        model_gb_1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_lr_1 = LinearRegression()

        model_rf_2 = RandomForestRegressor(n_estimators=100, random_state=42)
        model_gb_2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_lr_2 = LinearRegression()

        model_rf_3 = RandomForestRegressor(n_estimators=100, random_state=42)
        model_gb_3 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_lr_3 = LinearRegression()

        # Latih model untuk setiap angka
        model_rf_1.fit(X, y[:, 0])  # Angka pertama
        model_gb_1.fit(X, y[:, 0])
        model_lr_1.fit(X, y[:, 0])

        model_rf_2.fit(X, y[:, 1])  # Angka kedua
        model_gb_2.fit(X, y[:, 1])
        model_lr_2.fit(X, y[:, 1])

        model_rf_3.fit(X, y[:, 2])  # Angka ketiga
        model_gb_3.fit(X, y[:, 2])
        model_lr_3.fit(X, y[:, 2])

        # Kembalikan semua model yang sudah dilatih
        return (model_rf_1, model_gb_1, model_lr_1,
                model_rf_2, model_gb_2, model_lr_2,
                model_rf_3, model_gb_3, model_lr_3)
    except Exception as e:
        print(f"Gagal melatih model: {e}")
        return None

def prediksi_dengan_ensemble(models, input_terakhir):
    """
    Memprediksi 3 angka menggunakan model ensemble.
    Setiap angka diprediksi secara terpisah, lalu hasilnya digabungkan.
    """
    try:
        # Pisahkan model untuk setiap angka
        model_rf_1, model_gb_1, model_lr_1, \
        model_rf_2, model_gb_2, model_lr_2, \
        model_rf_3, model_gb_3, model_lr_3 = models

        # Prediksi setiap angka secara terpisah
        prediksi_1 = (model_rf_1.predict([input_terakhir]) +
                       model_gb_1.predict([input_terakhir]) +
                       model_lr_1.predict([input_terakhir])) / 3

        prediksi_2 = (model_rf_2.predict([input_terakhir]) +
                       model_gb_2.predict([input_terakhir]) +
                       model_lr_2.predict([input_terakhir])) / 3

        prediksi_3 = (model_rf_3.predict([input_terakhir]) +
                       model_gb_3.predict([input_terakhir]) +
                       model_lr_3.predict([input_terakhir])) / 3

        # Gabungkan prediksi untuk 3 angka
        prediksi_gabungan = np.array([prediksi_1, prediksi_2, prediksi_3])

        # Bulatkan dan batasi antara 1 dan 6
        return np.clip(np.round(prediksi_gabungan).astype(int), 1, 6)
    except Exception as e:
        print(f"Gagal melakukan prediksi: {e}")
        return None

def hapus_layar():
    """
    Membersihkan layar terminal.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("Program Tebak 3 Angka (1-6)")
    print("Masukkan 3 angka (pisahkan dengan spasi), atau ketik 'exit' untuk keluar.")
    print("Ketik 'hapus' untuk membersihkan layar.\n")

    while True:
        # Input dari pengguna
        user_input = input(">. Masukkan 3 angka yang keluar \n(angka1-6): ").strip().lower()

        # Keluar dari program jika pengguna mengetik 'exit'
        if user_input == "exit":
            print("Program selesai terimakasih")
            break

        # Membersihkan layar jika pengguna mengetik 'hapus'
        if user_input == "hapus":
            hapus_layar()
            continue

        # Validasi input
        if not validasi_input(user_input):
            print("Input tidak valid. Masukkan 3 angka antara 1 dan 6.")
            continue

        # Konversi input ke list angka
        numbers = list(map(int, user_input.split()))

        # Simpan input ke dataset
        simpan_data(numbers)

        # Latih model jika dataset cukup besar
        if len(dataset_inputs) >= 10:
            X = np.array(list(dataset_inputs)[:-1])  # Semua data kecuali yang terakhir
            y = np.array(list(dataset_outputs))  # Output adalah data berikutnya

            # Latih model ensemble
            models = latih_model_ensemble(X, y)

            # Prediksi nilai berikutnya
            if models is not None:
                input_terakhir = dataset_inputs[-1]
                prediksi = prediksi_dengan_ensemble(models, input_terakhir)
                if prediksi is not None:
                    jumlah = sum(prediksi)
                    kategori = "KECIL" if jumlah <= 10 else "BESAR"
                    print(f">. prediksi angka keluar berikutnya : {' '.join(map(str, prediksi))}")
                    print(f">. jumlah : {jumlah}")
                    print(f">. kategori : {kategori}")
                    print()
                else:
                    print("Belum bisa menebak. Terjadi kesalahan saat prediksi.")
            else:
                print("Belum bisa menebak. Terjadi kesalahan saat melatih model.")
        else:
            print("Belum bisa menebak. Butuh lebih banyak data.")

if __name__ == "__main__":
    main()