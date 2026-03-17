# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the programe

2.Import the required libraries such as pandas and sklearn.

3.Load the environmental sensor dataset from the CSV file.

4.Select Humidity, WindSpeed and Pressure as input features.

5.Select Temperature, PM2.5 and Energy as output variables.

6.Split the dataset into training and testing data.

7.Initialize the Random Forest Regressor model.

8.Train the model using the training dataset

9.Predict the output values using the test dataset

10.Display the predicted values.

Program:. 
Developed by: G sushanth
reg number:25011663

## Program:
```

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = {
    'Humidity': [60, 65, 70, 75, 80],
    'WindSpeed': [5, 7, 6, 8, 7],
    'Temperature': [30, 32, 31, 29, 28],
    'PM25': [120, 130, 125, 140, 150],
    'Energy': [200, 210, 205, 220, 230]
}

df = pd.DataFrame(data)

X = df[['Humidity', 'WindSpeed']]
y_temp = df['Temperature']
y_pm25 = df['PM25']
y_energy = df['Energy']

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_temp_train)
temp_pred = rf.predict(X_test)

rf.fit(X_train, y_pm25_train)
pm25_pred = rf.predict(X_test)

rf.fit(X_train, y_energy_train)
energy_pred = rf.predict(X_test)

print("Temperature MSE:", mean_squared_error(y_temp_test, temp_pred))
print("PM2.5 MSE:", mean_squared_error(y_pm25_test, pm25_pred))
print("Energy MSE:", mean_squared_error(y_energy_test, energy_pred))
```

## Output:
<img width="590" height="122" alt="image" src="https://github.com/user-attachments/assets/6327ee26-4782-4444-8b69-9efcd2a9131e" />


## Result:
