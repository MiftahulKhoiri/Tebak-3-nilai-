import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Variabel global untuk menyimpan dataset
dataset_inputs = deque(maxlen=100)  # Menyimpan input terakhir (maksimal 100 data)
dataset_outputs = deque(maxlen=100)  # Menyimpan output terakhir (maksimal 100 data)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))

def train_model():
    """
    Melatih model LSTM menggunakan data yang tersimpan.
    """
    if len(dataset_inputs) < 10:  # Minimal 10 data untuk melatih model
        return None

    # Konversi dataset ke numpy array
    X = np.array(list(dataset_inputs)[:-1])  # Semua data kecuali yang terakhir
    y = np.array(list(dataset_outputs))  # Output adalah data berikutnya

    # Normalisasi data
    X_normalized = scaler.fit_transform(X)
    y_normalized = scaler.fit_transform(y)

    # Ubah format data untuk LSTM (format: [samples, time steps, features])
    X_reshaped = X_normalized.reshape((X_normalized.shape[0], X_normalized.shape[1], 1))

    # Bangun model LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_reshaped.shape[1], 1)))  # 50 unit LSTM
    model.add(Dense(3))  # Output layer (3 angka)
    model.compile(optimizer='adam', loss='mse')

    # Latih model
    model.fit(X_reshaped, y_normalized, epochs=50, verbose=0)
    return model

def predict_next(model, last_input):
    """
    Memprediksi nilai berikutnya berdasarkan model LSTM dan input terakhir.
    """
    if model is None:
        return None

    # Normalisasi input terakhir
    last_input_normalized = scaler.transform([last_input])

    # Ubah format input untuk LSTM (format: [samples, time steps, features])
    last_input_reshaped = last_input_normalized.reshape((1, len(last_input), 1))

    # Prediksi nilai berikutnya
    prediction_normalized = model.predict(last_input_reshaped)

    # Denormalisasi prediksi
    prediction = scaler.inverse_transform(prediction_normalized)
    prediction = np.clip(np.round(prediction).astype(int), 1, 6)  # Bulatkan dan batasi antara 1 dan 6
    return prediction[0]

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