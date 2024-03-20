

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_squared_log_error


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Ambil output dari langkah terakhir saja
        return out
    
def predict_future_prices(model, scaler, initial_data, num_days=20):
    model.eval()
    with torch.no_grad():
        input_sequence = torch.tensor(initial_data).view(1, len(initial_data), 1).float()
        future_prices = []
        for _ in range(num_days):
            predicted_price = model(input_sequence).item()
            future_prices.append(predicted_price)
            input_sequence = torch.cat((input_sequence[:, 1:, :], torch.tensor([[predicted_price]]).view(1, 1, 1).float()), dim=1)

    future_prices_denormalized = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices_denormalized.flatten()


def get_data(start_date, end_date, csv_path):
    # Baca data dari file CSV
    data = pd.read_csv(csv_path)

    # Konversi kolom tanggal ke format datetime 
    if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
        data['Date'] = pd.to_datetime(data['Date'])
        
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data berdasarkan rentang tanggal
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    return filtered_data

# Menggunakan fungsi get_data untuk mendapatkan data dari file CSV
csv_path = 'BCH-USD.csv'  # Sesuaikan dengan lokasi file CSV Anda
start_date = '2017-12-16'
end_date = '2024-02-16'
stock_data = get_data(start_date, end_date, csv_path)


# Fungsi untuk melatih model
def train_model(data, seq_length=1000, num_epochs=500, hidden_size=35, num_layers=1, num_days_to_predict=5):
    # Preprocessing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Persiapkan data untuk pelatihan
    training_data = data['Close'].values
    seq_length = seq_length
    X_train, y_train = [], []
    for i in range(len(training_data) - seq_length):
        seq = training_data[i:i+seq_length]
        label = training_data[i+seq_length]
        X_train.append(seq)
        y_train.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Inisialisasi model LSTM
    input_size = 1
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    rmse_values = []
    
    train_data_scaled = scaler.transform(data['Close'].values.reshape(-1, 1))
    train_data_scaled = train_data_scaled.flatten()

    # Latih model
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Ambil data terakhir setelah pelatihan
        last_day_data_scaled = train_data_scaled[-seq_length:]
        last_day_data = scaler.inverse_transform(last_day_data_scaled.reshape(-1, 1)).flatten()
        
        # Prediksi harga beberapa hari ke depan
        predicted_prices = predict_future_prices(model, scaler, last_day_data, num_days=num_days_to_predict)
        actual_prices = data['Close'].values[-num_days_to_predict:]
        rmse_value = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        rmse_values.append(rmse_value)
        train_losses.append(loss.item())  # Ganti dengan nilai loss dari PyTorch

    return model, scaler, train_losses, rmse_values

# Streamlit App
def main():
    st.title("Bitcoin Cash (BCH) Price Prediction App")

    # Input parameter dari pengguna
    start_date = st.date_input("Pilih tanggal awal:", pd.to_datetime('2017-12-16'))
    end_date = st.date_input("Pilih tanggal akhir:", pd.to_datetime('2024-02-16'))
    seq_length = st.slider("Pilih panjang sekuens:", min_value=1, max_value=500, value=100)
    num_epochs = st.slider("Pilih jumlah epoch:", min_value=1, max_value=1000, value=500)
    hidden_size = st.slider("Pilih ukuran hidden layer:", min_value=1, max_value=1000, value=100)
    num_layers = st.slider("Pilih jumlah layer LSTM:", min_value=1, max_value=3, value=1)
    num_days_to_predict = st.slider("Pilih jumlah hari yang akan diprediksi:", min_value=1, max_value=365, value=1)

    # Dapatkan data sesuai parameter
    data = get_data(start_date, end_date, csv_path)

    # Latih model
    model, scaler, train_losses, rmse_values = train_model(data, seq_length=seq_length, num_epochs=num_epochs, hidden_size=hidden_size, num_layers=num_layers, num_days_to_predict=num_days_to_predict)


    # Hitung nilai MSE dan RMSE


    # Prediksi harga beberapa hari ke depan
    last_day_data = data['Close'].values[-seq_length:]
    predicted_future_prices = predict_future_prices(model, scaler, last_day_data, num_days=num_days_to_predict)
   
    #menentukan nilai rmse
    actual_prices = data['Close'].values[-num_days_to_predict:]
    rmse_value = np.sqrt(mean_squared_error(actual_prices, predicted_future_prices))

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi Harga BCH:")
    predicted_df = pd.DataFrame({
    'Tanggal': pd.date_range(start=end_date + pd.DateOffset(1), periods=num_days_to_predict),
    'Prediksi Harga': predicted_future_prices
    })

    fig = px.line(predicted_df, x='Tanggal', y='Prediksi Harga', title='Bitcoin Cash (BCH) Price Prediction')
    fig.update_layout(xaxis_title='Date', yaxis_title='Predicted Close Price (USD)')

    st.plotly_chart(fig)
    st.write(predicted_df)

    # Tampilkan grafik fungsi loss
    st.subheader("Grafik Nilai RMSE")
    rmse_df = pd.DataFrame({'Epoch': range(1, num_epochs+1), 'RMSE': rmse_values})
    fig_rmse = px.line(rmse_df, x='Epoch', y='RMSE', title='RMSE Over Epochs')
    fig_rmse.update_layout(xaxis_title='Epoch', yaxis_title='RMSE')
    st.plotly_chart(fig_rmse)
    
    st.subheader("Grafik Fungsi Loss (MSE)")
    loss_df = pd.DataFrame({'Epoch': range(1, num_epochs+1), 'MSE Loss': train_losses})
    fig_loss = px.line(loss_df, x='Epoch', y='MSE Loss', title='MSE Loss Over Epochs')
    fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='MSE Loss')

    st.plotly_chart(fig_loss)




if __name__ == "__main__":
    main()


















