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

# Variabel global untuk menghitung prediksi berhasil dan gagal
prediksi_berhasil = 0
prediksi_gagal = 0

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
    global rata_rata_kesalahan, prediksi_berhasil, prediksi_gagal

    # Tentukan rentang nilai aktual
    if 3 <= nilai_aktual <= 10:
        rentang_aktual = "KECIL"
    elif 11 <= nilai_aktual <= 18:
        rentang_aktual = "BESAR"
    else:
        prediksi_gagal += 1  # Tambahkan ke prediksi gagal
        return "Gagal"  # Nilai aktual di luar rentang yang valid

    # Tentukan rentang nilai prediksi
    if 3 <= nilai_prediksi <= 10:
        rentang_prediksi = "KECIL"
    elif 11 <= nilai_prediksi <= 18:
        rentang_prediksi = "BESAR"
    else:
        prediksi_gagal += 1  # Tambahkan ke prediksi gagal
        return "Gagal"  # Nilai prediksi di luar rentang yang valid

    # Jika rentang aktual dan prediksi berbeda, evaluasi adalah Gagal
    if rentang_aktual != rentang_prediksi:
        prediksi_gagal += 1  # Tambahkan ke prediksi gagal
        return "Gagal"

    # Hitung selisih antara nilai aktual dan prediksi
    selisih = abs(nilai_prediksi - nilai_aktual)

    # Evaluasi berdasarkan selisih
    if selisih == 0:
        prediksi_berhasil += 1  # Tambahkan ke prediksi berhasil
        return "Sempurna"
    elif selisih == 1:
        prediksi_berhasil += 1  # Tambahkan ke prediksi berhasil
        return "Bagus"
    elif selisih == 2:
        prediksi_berhasil += 1  # Tambahkan ke prediksi berhasil
        return "Kurang Bagus"
    elif selisih == 3:
        prediksi_berhasil += 1  # Tambahkan ke prediksi berhasil
        return "Buruk"
    else:
        prediksi_gagal += 1  # Tambahkan ke prediksi gagal
        return "Gagal"  # Selisih lebih dari 3

def perbaiki_prediksi(prediksi, evaluasi_terakhir):
    """
    Menyesuaikan prediksi berdasarkan evaluasi terakhir.
    Jika evaluasi terakhir adalah "Bagus" atau "Sempurna", prediksi tidak diubah.
    Jika evaluasi terakhir adalah "Kurang Bagus", "Buruk", atau "Gagal", prediksi disesuaikan.
    """
    global rata_rata_kesalahan

    # Jika evaluasi terakhir adalah "Bagus" atau "Sempurna", tidak perlu perbaikan
    if evaluasi_terakhir in ["Bagus", "Sempurna"]:
        return prediksi

    # Jika rata-rata kesalahan besar, lakukan perbaikan
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
    Menjalankan demo dengan jumlah iterasi yang ditentukan oleh pengguna.dan waktu jeda untuk melihat hasil prediksi.
    """
    print("\nMemulai demo...\n")

    # Minta input jumlah demo dari pengguna
    while True:
        try:
            jumlah_demo = int(input("Masukkan jumlah demo (minimal 11): "))
            waktu_tampil = int(input("Berapa lama ingin melihat hasil! (detik): "))
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
        print(f"\n Demo [{i+1}/{jumlah_demo}]\n >. Memasukkan angka acak :[{input_demo}]")

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
                    jumlah_aktual = sum(dataset_outputs[-1])
                    jumlah_prediksi = sum(prediksi)
                    evaluasi = evaluasi_prediksi(jumlah_aktual, jumlah_prediksi)
                    prediksi = perbaiki_prediksi(prediksi, evaluasi)  # Perbaiki prediksi berdasarkan evaluasi
                    jumlah_prediksi = sum(prediksi)
                    kategori_aktual = "KECIL" if 3 <= jumlah_aktual <= 10 else "BESAR"
                    kategori_prediksi = "KECIL" if 3 <= jumlah_prediksi <= 10 else "BESAR"

                    # Evaluasi prediksi
                    print("-"*35)
                    print(f">. evaluasi prediksi : {evaluasi}")
                    print(f">. rata-rata kesalahan : {rata_rata_kesalahan:.2f}")
                    print(f">. Prediksi berhasil: {prediksi_berhasil} kali")
                    print(f">. Prediksi gagal: {prediksi_gagal} kali\n")

                    print(f">. prediksi angka keluar berikutnya : {' '.join(map(str, prediksi))}")
                    print(f">. jumlah prediksi : {jumlah_prediksi} ({kategori_prediksi})")
                    print(f">. Nilai terbaru yang keluar : {input_demo}")
                    print(f">. jumlah aktual : {jumlah_aktual} ({kategori_aktual})")


                    print("-"*35)
                else:
                    print("\n Belum bisa menebak. \n Terjadi kesalahan saat prediksi.\n")
            else:
                print("\n Belum bisa menebak. \n Terjadi kesalahan saat melatih model.\n")
        else:
            print("\n Belum bisa menebak. \n Butuh lebih banyak data lagi!\n")

        # Jeda sesuai waktu yang diminta
        time.sleep(waktu_tampil)
        hapus_layar()

def prediksi():
    """
    Fungsi untuk melakukan prediksi berdasarkan input pengguna.
    """
    print("\nMemulai prediksi...\n")
    while True:
        user_input = input(">.Masukkan 3 angka yang keluar \n (angka1-6): ").strip().lower()

        if user_input == "exit" or user_input == "selesai":
            print("Keluar dari mode prediksi.")
            break

        if user_input == "hapus":
            hapus_layar()
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
                    jumlah_aktual = sum(dataset_outputs[-1])
                    jumlah_prediksi = sum(prediksi)
                    evaluasi = evaluasi_prediksi(jumlah_aktual, jumlah_prediksi)
                    prediksi = perbaiki_prediksi(prediksi, evaluasi)  # Perbaiki prediksi berdasarkan evaluasi
                    jumlah_prediksi = sum(prediksi)
                    kategori_aktual = "KECIL" if 3 <= jumlah_aktual <= 10 else "BESAR"
                    kategori_prediksi = "KECIL" if 3 <= jumlah_prediksi <= 10 else "BESAR"

                    # Evaluasi prediksi
                    print("-"*35)
                    print(f">. evaluasi prediksi : {evaluasi}")
                    print(f">. rata-rata kesalahan : {rata_rata_kesalahan:.2f}")
                    print(f">. Prediksi berhasil: {prediksi_berhasil} kali")
                    print(f">. Prediksi gagal: {prediksi_gagal} kali\n")
                    print(f">. Nilai terbaru yang keluar : {user_input}")
                    print(f">. jumlah aktual : {jumlah_aktual} ({kategori_aktual})")
                    print(f">. prediksi angka keluar berikutnya : {' '.join(map(str, prediksi))}")
                    print(f">. Jumlah prediksi: {jumlah_prediksi}({kategori_prediksi})")

                    print("-"*35)
                else:
                    print("\n Belum bisa menebak. Terjadi kesalahan saat prediksi.\n")
            else:
                print("\n Belum bisa menebak. Terjadi kesalahan saat melatih model.\n")
        else:
            print("\n Belum bisa menebak. Butuh lebih banyak data lagi!\n")

def main():
    """
    Fungsi utama untuk menampilkan menu awal dan memilih antara prediksi atau demo.
    """
    print("\t # Program Prediksi 3 Angka #")
    

    while True:
     print(" Pilih opsi:")
     print(" 1. Prediksi")
     print(" 2. Demo")
     print(" 3. Exit")
     pilihan = input(">.Masukkan pilihan (1/2/3): ").strip()
     if pilihan == "1":
            prediksi()
     elif pilihan == "2":
            demo()
     elif pilihan == "3":
            print("Program selesai. Terima kasih!")
            break
     else:
            print("Pilihan tidak valid. Silakan masukkan 1, 2, atau 3.")

if __name__ == "__main__":
    main()