import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from collections import deque
import os
import logging
import time  # Untuk jeda
import random  # Untuk menghasilkan angka acak

# Setup logging untuk mencatat pesan error atau info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variabel global untuk menyimpan dataset
dataset_inputs = deque(maxlen=1000)  # Menyimpan input terakhir (maksimal 1000 data)
dataset_outputs = deque(maxlen=1000)  # Menyimpan output terakhir (maksimal 1000 data)

# Variabel global untuk menyimpan riwayat evaluasi dan rata-rata kesalahan
riwayat_evaluasi = []
rata_rata_kesalahan = 0

def validasi_input(user_input):
    """
    Memvalidasi input pengguna. Input harus berupa 3 angka antara 1 dan 6.
    """
    try:
        numbers = list(map(int, user_input.strip().split()))
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
        logging.error(f"Gagal menyimpan data: {e}")

def latih_model_ensemble(X, y):
    """
    Melatih model ensemble untuk setiap angka (angka pertama, angka kedua, angka ketiga).
    Setiap angka diprediksi secara terpisah menggunakan 3 model: Random Forest, Gradient Boosting, dan Linear Regression.
    """
    try:
        models = []
        for _ in range(3):  # Untuk setiap angka (3 angka)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            lr = LinearRegression()
            models.append((rf, gb, lr))  # Simpan model untuk satu angka

        # Latih model untuk setiap angka
        for i in range(3):  # Untuk setiap angka
            for model in models[i]:
                model.fit(X, y[:, i])  # Latih model untuk angka ke-i

        return models
    except Exception as e:
        logging.error(f"Gagal melatih model: {e}")
        return None

def prediksi_dengan_ensemble(models, input_terakhir):
    """
    Memprediksi 3 angka menggunakan model ensemble.
    Setiap angka diprediksi secara terpisah, lalu hasilnya digabungkan.
    """
    try:
        prediksi = []
        for i in range(3):  # Untuk setiap angka
            prediksi_angka = np.mean([model.predict([input_terakhir]) for model in models[i]], axis=0)
            prediksi.append(prediksi_angka)

        prediksi_gabungan = np.array(prediksi).flatten()
        return np.clip(np.round(prediksi_gabungan).astype(int), 1, 6)
    except Exception as e:
        logging.error(f"Gagal melakukan prediksi: {e}")
        return None

def evaluasi_prediksi(nilai_aktual, nilai_prediksi):
    """
    Mengevaluasi prediksi berdasarkan rentang kecil (3-10) atau besar (11-18).
    - Sempurna: Prediksi sama dengan nilai aktual.
    - Bagus: Prediksi ±1 dari nilai aktual.
    - Kurang Bagus: Prediksi ±2 dari nilai aktual.
    - Buruk: Prediksi ±3 dari nilai aktual.
    - Gagal: Prediksi di luar rentang atau selisih > 3.
    """
    global rata_rata_kesalahan

    # Tentukan rentang nilai aktual
    if 3 <= nilai_aktual <= 10:
        rentang_aktual = "KECIL"
    elif 11 <= nilai_aktual <= 18:
        rentang_aktual = "BESAR"
    else:
        return "Gagal"  # Nilai aktual di luar rentang yang valid

    # Tentukan rentang nilai prediksi
    if 3 <= nilai_prediksi <= 10:
        rentang_prediksi = "KECIL"
    elif 11 <= nilai_prediksi <= 18:
        rentang_prediksi = "BESAR"
    else:
        return "Gagal"  # Nilai prediksi di luar rentang yang valid

    # Jika rentang aktual dan prediksi berbeda, evaluasi adalah Gagal
    if rentang_aktual != rentang_prediksi:
        return "Gagal"

    # Hitung selisih antara nilai aktual dan prediksi
    selisih = abs(nilai_aktual - nilai_prediksi)

    # Evaluasi berdasarkan selisih
    if selisih == 0:
        return "Sempurna"
    elif selisih == 1:
        return "Bagus"
    elif selisih == 2:
        return "Kurang Bagus"
    elif selisih == 3:
        return "Buruk"
    else:
        return "Gagal"  # Selisih lebih dari 3

def perbaiki_prediksi(prediksi):
    """
    Menyesuaikan prediksi berdasarkan rata-rata kesalahan.
    """
    global rata_rata_kesalahan
    if rata_rata_kesalahan > 1:  # Jika rata-rata kesalahan besar, lakukan perbaikan
        prediksi = np.clip(prediksi + np.random.randint(-1, 2, size=3), 1, 6)  # Sesuaikan prediksi
    return prediksi

def hapus_layar():
    """
    Membersihkan layar terminal.
    """
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception as e:
        logging.error(f"Gagal membersihkan layar: {e}")

def demo():
    """
    Menjalankan demo dengan jumlah iterasi yang ditentukan oleh pengguna.
    """
    print("\nMemulai demo...\n")
    
    # Minta input jumlah demo dari pengguna
    while True:
        try:
            jumlah_demo = int(input("Masukkan jumlah demo (minimal 11): "))
            if jumlah_demo >= 11:
                break
            else:
                print("Jumlah demo minimal 11. Silakan coba lagi.")
        except ValueError:
            print("Input tidak valid. Masukkan angka.")

    print(f"\nMenjalankan {jumlah_demo} iterasi demo...\n")
    for i in range(jumlah_demo):  # Jumlah iterasi sesuai input pengguna
        # Generate 3 angka acak antara 1 dan 6
        angka_acak = [random.randint(1, 6) for _ in range(3)]
        input_demo = ' '.join(map(str, angka_acak))
        print(f"\nDemo {i+1}/{jumlah_demo}: Memasukkan angka acak :[{input_demo}]")
        
        # Simulasikan input pengguna
        numbers = list(map(int, input_demo.split()))
        simpan_data(numbers)

        if len(dataset_inputs) >= 10:
            X = np.array(list(dataset_inputs)[:-1])  # Semua data kecuali yang terakhir
            y = np.array(list(dataset_outputs))  # Output adalah data berikutnya

            models = latih_model_ensemble(X, y)
            if models is not None:
                input_terakhir = dataset_inputs[-1]
                prediksi = prediksi_dengan_ensemble(models, input_terakhir)
                if prediksi is not None:
                    prediksi = perbaiki_prediksi(prediksi)  # Perbaiki prediksi berdasarkan evaluasi
                    jumlah_aktual = sum(dataset_outputs[-1])
                    jumlah_prediksi = sum(prediksi)
                    kategori_aktual = "KECIL" if 3 <= jumlah_aktual <= 10 else "BESAR"
                    kategori_prediksi = "KECIL" if 3 <= jumlah_prediksi <= 10 else "BESAR"

                    # Evaluasi prediksi
                    evaluasi = evaluasi_prediksi(jumlah_aktual, jumlah_prediksi)
                    print("-"*35)
                    print(f">. evaluasi prediksi : {evaluasi}")
                    print(f">. rata-rata kesalahan : {rata_rata_kesalahan:.2f}\n")
                    print(f">. Nilai terbaru yang keluar : {input_demo}")
                    print(f">. jumlah aktual : {jumlah_aktual} ({kategori_aktual})")
                    print(f">. prediksi angka keluar berikutnya : {' '.join(map(str, prediksi))}")
                    print(f">. jumlah prediksi : {jumlah_prediksi} ({kategori_prediksi})")
                    print("-"*35)
                else:
                    print("\n Belum bisa menebak. Terjadi kesalahan saat prediksi.\n")
            else:
                print("\n Belum bisa menebak. Terjadi kesalahan saat melatih model.\n")
        else:
            print("\n Belum bisa menebak. Butuh lebih banyak data lagi!\n")

        # Jeda 5 detik
        time.sleep(5)
        hapus_layar()

def main():
    print("\t # Program Prediksi 3 Angka #")
    print(" Masukkan 3 angka (pisahkan dengan spasi).\n Ketik 'hapus' untuk membersihkan layar.\n Ketik 'demo' untuk menjalankan demo.\n atau ketik 'exit/selesi' untuk keluar.\n")

    while True:
        user_input = input(">.Masukkan 3 angka yang keluar \n (angka1-6): ").strip().lower()

        if user_input == "exit" or user_input == "selesai":
            print("Program selesai. Terima kasih!")
            break

        if user_input == "hapus":
            hapus_layar()
            continue

        if user_input == "demo":
            demo()
            continue

        if not validasi_input(user_input):
            print("Input tidak valid. Masukkan 3 angka antara 1 dan 6, dipisahkan dengan spasi.")
            continue

        numbers = list(map(int, user_input.split()))
        simpan_data(numbers)

        if len(dataset_inputs) >= 10:
            X = np.array(list(dataset_inputs)[:-1])  # Semua data kecuali yang terakhir
            y = np.array(list(dataset_outputs))  # Output adalah data berikutnya

            models = latih_model_ensemble(X, y)
            if models is not None:
                input_terakhir = dataset_inputs[-1]
                prediksi = prediksi_dengan_ensemble(models, input_terakhir)
                if prediksi is not None:
                    prediksi = perbaiki_prediksi(prediksi)  # Perbaiki prediksi berdasarkan evaluasi
                    jumlah_aktual = sum(dataset_outputs[-1])
                    jumlah_prediksi = sum(prediksi)
                    kategori_aktual = "KECIL" if 3 <= jumlah_aktual <= 10 else "BESAR"
                    kategori_prediksi = "KECIL" if 3 <= jumlah_prediksi <= 10 else "BESAR"

                    # Evaluasi prediksi
                    evaluasi = evaluasi_prediksi(jumlah_aktual, jumlah_prediksi)
                    print("-"*35)
                    print(f">. evaluasi prediksi : {evaluasi}")
                    print(f">. rata-rata kesalahan : {rata_rata_kesalahan:.2f}\n")
                    print(f">. Nilai terbaru yang keluar : {user_input}")
                    print(f">. jumlah aktual : {jumlah_aktual} ({kategori_aktual})")
                    print(f">. prediksi angka keluar berikutnya : {' '.join(map(str, prediksi))}")
                    print(f">. jumlah prediksi : {jumlah_prediksi} ({kategori_prediksi})")
                    print("-"*35)
                else:
                    print("\n Belum bisa menebak. Terjadi kesalahan saat prediksi.\n")
            else:
                print("\n Belum bisa menebak. Terjadi kesalahan saat melatih model.\n")
        else:
            print("\n Belum bisa menebak. Butuh lebih banyak data lagi!\n")

if __name__ == "__main__":
    main()#