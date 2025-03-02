import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque

# Variabel global untuk menyimpan dataset
dataset_inputs = deque(maxlen=100)  # Menyimpan input terakhir (maksimal 100 data)
dataset_outputs = deque(maxlen=100)  # Menyimpan output terakhir (maksimal 100 data)

def train_model():
    """
    Melatih model menggunakan data yang tersimpan.
    """
    if len(dataset_inputs) < 10:  # Minimal 10 data untuk melatih model
        return None

    # Konversi dataset ke numpy array
    X = np.array(list(dataset_inputs)[:-1])  # Semua data kecuali yang terakhir
    y = np.array(list(dataset_outputs))  # Output adalah data berikutnya

    # Buat dan latih model regresi linier
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_next(model, last_input):
    """
    Memprediksi nilai berikutnya berdasarkan model dan input terakhir.
    """
    if model is None:
        return None
    # Prediksi nilai berikutnya
    prediction = model.predict([last_input])
    return np.clip(np.round(prediction).astype(int), 1, 6)  # Bulatkan dan batasi antara 1 dan 6

def main():
    print("Program Tebak 3 Angka (1-6)")
    print("Masukkan 3 angka (pisahkan dengan spasi), atau ketik 'exit' untuk keluar.")

    while True:
        # Input dari pengguna
        user_input = input("Masukkan 3 angka (1-6): ").strip().lower()

        # Keluar dari program jika pengguna mengetik 'exit'
        if user_input == "exit":
            print("Program dihentikan.")
            break

        # Validasi input
        try:
            numbers = list(map(int, user_input.split()))
            if len(numbers) != 3 or any(num < 1 or num > 6 for num in numbers):
                print("Input tidak valid. Masukkan 3 angka antara 1 dan 6.")
                continue
        except ValueError:
            print("Input tidak valid. Masukkan 3 angka antara 1 dan 6.")
            continue

        # Simpan input ke dataset
        if len(dataset_inputs) > 0:
            dataset_outputs.append(numbers)  # Output adalah input berikutnya
        dataset_inputs.append(numbers)  # Input adalah data sebelumnya

        # Latih model
        model = train_model()

        # Prediksi nilai berikutnya
        if model is not None and len(dataset_inputs) > 1:
            last_input = dataset_inputs[-1]
            prediction = predict_next(model, last_input)
            if prediction is not None:
                print(f"Prediksi angka berikutnya: {prediction}")
            else:
                print("Belum bisa menebak. Butuh lebih banyak data.")
        else:
            print("Belum bisa menebak. Butuh lebih banyak data.")

if __name__ == "__main__":
    main()