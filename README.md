# EX-6 Implement of Holt Winters method in Python
## NAME : HARIHARAN J
## REG NO : 212223240047

# ALGORITHM:

1.Load the dataset and set the date column as the index.

2.Resample data to monthly frequency for seasonality analysis.

3.Split data into training (80%) and testing (20%).

4.Fit Holt-Winters model with additive trend, multiplicative seasonality, and seasonal period = 12.

5.Forecast values for the test set and future periods.

6.Evaluate and visualize results using RMSE and forecast plots.


# PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('Crypto Data Since 2015 (1).csv',parse_dates=['Date'],index_col='Date')

data.head()

data_monthly = data.resample('MS').sum()

data_monthly.head()

data_monthly.plot()

data.plot()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly['Bitcoin (USD)'].values.reshape(-1, 1)).flatten(),index=data_monthly.index)
scaled_data.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly['Bitcoin (USD)'], model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()

final_model = ExponentialSmoothing(data_monthly['Bitcoin (USD)'],trend='add',seasonal='mul',seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))
ax = data_monthly.plot(figsize=(10, 6))
final_predictions.plot(ax=ax, color='red')
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Years')
ax.set_ylabel('Price of the Bitcoin')
ax.set_title('Exponential Smoothing Forecast')
plt.show()

```
# OUTPUT:

Scaled_data Plot

<img width="547" height="432" alt="image" src="https://github.com/user-attachments/assets/170f5c2a-47c2-498b-8933-ea6db5569714" />

Decomposed Plot

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/b91ff763-d43d-4660-94c3-66f6564d4b9c" />

Test Predictions

<img width="547" height="455" alt="image" src="https://github.com/user-attachments/assets/22b9af06-02a6-4777-b623-b10d90a26c6d" />

Final Predictions

<img width="857" height="547" alt="image" src="https://github.com/user-attachments/assets/1e1dbafe-6835-4568-8cf9-84b7cf606180" />

# RESULT:

Thus, The Holt-Winters method was successfully implemented in Python.
