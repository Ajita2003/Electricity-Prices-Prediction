#import matplotlib
#matplotlib.use('Agg')  # Use the non-GUI backend 'Agg'
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

app = Flask(__name__)

# Data Loading
data = pd.read_csv("Electricity.csv", low_memory=False)

# Data Preprocessing
cols = ["ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2", "SMPEP2"]
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

scaler = StandardScaler()
data[cols[:-1]] = scaler.fit_transform(data[cols[:-1]])

x = data[cols[:-1]]
y = data["SMPEP2"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(xtrain, ytrain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(request.form[col]) for col in cols[:-1]]
    features_scaled = scaler.transform([inputs])
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
    """features = cols[:-1] + ["SMPEP2"]
    correlations = data[features].corr(method='pearson')
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, cmap="coolwarm", annot=True)
    plot_path = os.path.join('static', 'plots', 'correlation_plot.png')
    plt.savefig(plot_path)
    plt.close()"""
    return send_from_directory('static/plots', 'correlation_plot.png')

@app.route('/month_vs_smpep2_plot')
def month_vs_smpep2_plot():
    """plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="Month", y="SMPEP2")
    plt.xlabel("Month")
    plt.ylabel("SMPEP2")
    plt.title("Month Vs SMPEP2")
    plot_path = os.path.join('static', 'plots', 'month_vs_smpep2_plot.png')
    plt.savefig(plot_path)
    plt.close()"""
    return send_from_directory('static/plots', 'month_vs_smpep2_plot.png')

@app.route('/static/plots/<filename>')
def send_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == "__main__":
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    app.run(debug=True)
