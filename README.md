# Time Series Prediction Web Application

This web application performs state-of-the-art Time Series Prediction using three advanced algorithms:

- **TimesNet** :  A time series network which fourier transform to transform 1-D data to 2-D data which disentangle the intraperiod- and interperiod-variations lie in the time series and utilize Inception Network as learning algorithm.
- **XLSTM** : A new version of LSTM which combines blocks of sLSTM(scalar memory, new memory mixing such as using exponential as activation function, perform stabilization) and mLSTM(fully parallelizable with a
matrix memory and a covariance update rule).
- **ITransformer** : Instead of using a scalar representation for each variates in the time series, vector representation is employed using a projection layer and pass to transformer layer.

These cutting-edge models have been trained on diverse datasets, including:

- **Water Potability Data**: Predicts the potability of water based on several chemical characteristics.
- **Rainfall Data (India)**: Forecasts the next 12 months rainfall amounts given the last 5 years historical data.
- **Weather Data**: Predicts the next day minimum, maximum and average temperature given the last 31 days (window size) data for major cities: Beijing, California, London, Tokyo, and Singapore. This data was collected using the Meteostat API.

## Key Features

- **Rainfall Predictions**: From 2016 to 2028, this feature predicts monthly rainfall based on the last five years of data for each month.
- **Weather Forecasts**: Forecasts daily weather (minimum, average, and maximum temperature) from 2024 to 2030 using the last 31 days of data, including factors like date, month, year, and city.
- **Binary Classification on Water Potability**: This feature classifies water as potable or non-potable based on its chemical properties.

### Data Sources

- **data.world**: Used to gather data on water potability and Indian rainfall records.
- **Meteostat API**: Provides comprehensive weather data for five major cities across the world, including Beijing, California, London, Tokyo, and Singapore.

## Explore the Application

The application allows users to perform predictions using the aforementioned algorithms. By providing input data, you can explore our predictions and experience the power of advanced time series analysis.

### Access the Application

You can access the web application here: [Time Series Prediction App](https://nunatimeseriesprediction.streamlit.app/).
