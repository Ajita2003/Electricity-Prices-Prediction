from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from io import BytesIO
import base64

app = Flask(__name__)

# Data Loading
data = pd.read_csv("Electricity.csv", low_memory=False)

# Data Preprocessing
data["ForecastWindProduction"] = pd.to_numeric(data["ForecastWindProduction"], errors='coerce')
data["SystemLoadEA"] = pd.to_numeric(data["SystemLoadEA"], errors='coerce')
data["SMPEA"] = pd.to_numeric(data["SMPEA"], errors='coerce')
data["ORKTemperature"] = pd.to_numeric(data["ORKTemperature"], errors='coerce')
data["ORKWindspeed"] = pd.to_numeric(data["ORKWindspeed"], errors='coerce')
data["CO2Intensity"] = pd.to_numeric(data["CO2Intensity"], errors='coerce')
data["ActualWindProduction"] = pd.to_numeric(data["ActualWindProduction"], errors='coerce')
data["SystemLoadEP2"] = pd.to_numeric(data["SystemLoadEP2"], errors='coerce')
data["SMPEP2"] = pd.to_numeric(data["SMPEP2"], errors='coerce')

scaler = StandardScaler()
data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2"]] = scaler.fit_transform(
    data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2"]])

data = data.dropna()

x = data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2"]]
y = data["SMPEP2"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(xtrain, ytrain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Day = int(request.form['Day'])
    Month = int(request.form['Month'])
    FWP = float(request.form['ForecastWindProduction'])
    SLE = float(request.form['SystemLoadEA'])
    SMP = float(request.form['SMPEA'])
    ORKT = float(request.form['ORKTemperature'])
    ORKW = float(request.form['ORKWindspeed'])
    CO2 = float(request.form['CO2Intensity'])
    Actualwind = float(request.form['ActualWindProduction'])
    SLE2 = float(request.form['SystemLoadEP2'])
    
    features = np.array([[Day, Month, FWP, SLE, SMP, ORKT, ORKW, CO2, Actualwind, SLE2]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return render_template('index.html', prediction=prediction)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    actual = float(request.form['actual'])
    predicted = float(request.form['predicted'])

    mae = mean_absolute_error([actual], [predicted])
    mse = mean_squared_error([actual], [predicted])
    rmse = np.sqrt(mse)

    return render_template('index.html', mae=mae, mse=mse, rmse=rmse)

@app.route('/correlation_plot')
def correlation_plot():
    # Select features for correlation calculation
    features = [
        "ForecastWindProduction", "SystemLoadEA", "SMPEA",
        "ORKTemperature", "ORKWindspeed", "CO2Intensity",
        "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
    ]

    # Calculate correlations and create a heatmap
    correlations = data[features].corr(method='pearson')
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, cmap="coolwarm", annot=True)
    plot_path = os.path.join('static', 'plots', 'correlation_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    return render_template('index.html', correlation_plot='correlation_plot.png')

@app.route('/month_vs_smpep2_plot')
def month_vs_smpep2_plot():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="Month", y="SMPEP2")
    plt.xlabel("Month")
    plt.ylabel("SMPEP2")
    plt.title("Month Vs SMPEP2")
    plot_path = os.path.join('static', 'plots', 'month_vs_smpep2_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    return render_template('index.html', month_vs_smpep2_plot='month_vs_smpep2_plot.png')

@app.route('/static/plots/<filename>')
def send_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == "__main__":
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    app.run(debug=True)
