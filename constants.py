import os

BASE_PATH = os.getcwd()
WEIGHT_PATH = BASE_PATH + "/weights"
UTILITY_PATH = BASE_PATH + "/utility"
PREDICTION_PATH = BASE_PATH + "/prediction"

XLSTM_RAINFALL = WEIGHT_PATH + "/xlstm_rainfall.pth"
XLSTM_WATER = WEIGHT_PATH + "/xlstm_water.pth"
XLSTM_WEATHER = WEIGHT_PATH + "/xlstm_weather.pth"
ITRANSFORMER_RAINFALL = WEIGHT_PATH + "/iTransformer_rainfall.pth"
ITRANSFORMER_WATER = WEIGHT_PATH + "/iTransformer_water.pth"
ITRANSFORMER_WEATHER = WEIGHT_PATH + "/iTransformer_weather.pth"
TIMESNET_RAINFALL = WEIGHT_PATH + "/Timesnet_rainfall.pth"
TIMESNET_WATER = WEIGHT_PATH + "/Timesnet_water.pth"
TIMESNET_WEATHER = WEIGHT_PATH + "/Timesnet_weather.pth"

SCALER = UTILITY_PATH + "/rainfall_scaler.pkl"

XLSTM_RAINFALL_PREDICTION = PREDICTION_PATH + "/xlstm_rainfall.csv"
XLSTM_WEATHER_PREDICTION  = PREDICTION_PATH + "/xlstm_weather.csv"
ITRANSFORMER_RAINFALL_PREDICTION  = PREDICTION_PATH + "/itransformer_rainfall.csv"
ITRANSFORMER_WEATHER_PREDICTION  = PREDICTION_PATH + "/itransformer_weather.csv"
TIMESNET_RAINFALL_PREDICTION  = PREDICTION_PATH + "/timesnet_rainfall.csv"
TIMESNET_WEATHER_PREDICTION  = PREDICTION_PATH + "/timesnet_weather.csv"
