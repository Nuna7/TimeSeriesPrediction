import torch
import streamlit as st
import joblib

from models import (
    xlstm_weather, 
    xlstm_water, 
    xlstm_rainfall, 
    water_itransformer, 
    rainfall_itransformer,
    weather_itransformer,
    timesnet_rainfall,
    timesnet_water,
    timesnet_weather,
    subdivision_to_idx,
    cities_to_idx
)

from constants import (
    SCALER,
    XLSTM_RAINFALL_PREDICTION,
    XLSTM_WEATHER_PREDICTION,
    ITRANSFORMER_RAINFALL_PREDICTION,
    ITRANSFORMER_WEATHER_PREDICTION,
    TIMESNET_RAINFALL_PREDICTION,
    TIMESNET_WEATHER_PREDICTION
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

rainfall_scaler = joblib.load(SCALER)

st.set_page_config(
    page_title="Time Series Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="collapsed",
)


page_bg = '''
    <style>
    .stApp {
        background-size: cover;
        background-position: center;
    }
    .big-font {
        font-size:30px !important;
        padding-bottom: 20px;
    }
    .highlight {
        padding: 10px;
        border-radius: 5px;
    }
    </style>
'''

st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("""
    <div class="big-font">
    Welcome to my Time Series Prediction Application!
    </div>
    """, unsafe_allow_html=True)

if 'selected' not in st.session_state:
    st.session_state.selected = None

if st.session_state.selected is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Water"):
            st.session_state.selected = 'Water'
            st.rerun()

    with col2:
        if st.button("Rainfall"):
            st.session_state.selected = 'Rainfall'
            st.rerun()

    with col3:
        if st.button("Weather"):
            st.session_state.selected = 'Weather'
            st.rerun()

    st.markdown("""
    <div class="highlight">
    This web application performs state-of-the-art Time Series Prediction using:
    <ul>
        <li>TimesNet</li>
        <li>XLSTM</li>
        <li>ITransformer</li>
    </ul>
    
    These advanced algorithms have been trained on diverse datasets:
    <ul>
        <li>Water potability data</li>
        <li>Rainfall data (India)</li>
        <li>Weather data from 5 major cities: Beijing, California, London, Tokyo, and Singapore</li>
    </ul>
    
    Our data sources include:
    <ul>
        <li><a href="https://data.world/" target="_blank">data.world</a> for water potability and rainfall data</li>
        <li>Meteostat API for comprehensive weather data</li>
    </ul>
    
    Key Features:
    <ul>
        <li>Rainfall predictions from 2016 to 2028</li>
        <li>Weather forecasts from 2024 to 2030</li>
        <li>Perform prediction using the given algorithms by providing input data</li>
    </ul>
    
    Explore our predictions and discover the power of advanced time series analysis!
    </div>
    """, unsafe_allow_html=True)
else:
    if st.session_state.selected == 'Water':
        st.markdown("<h3 style='text-align: center;'>Water Potability Prediction</h3>", unsafe_allow_html=True)

        with st.form(key='water_form'):
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
            hardness = st.number_input("Hardness", min_value=0.0)
            solids = st.number_input("Solids", min_value=0.0)
            chloramines = st.number_input("Chloramines", min_value=0.0)
            conductivity = st.number_input("Conductivity", min_value=0.0)
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
            turbidity = st.number_input("Turbidity", min_value=0.0)

            selected_models = st.multiselect("Select up to 3 models", ['XLSTM', 'ITRANSFORMER', 'TIMESNET'], max_selections=3)

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            input_data = [ph, hardness, solids, chloramines, conductivity, organic_carbon, trihalomethanes, turbidity]
            input_data = torch.Tensor(input_data).view(1, 8, 1)
            model_outputs = {}

            if 'XLSTM' in selected_models:
                model_outputs['XLSTM'] = xlstm_water.forward(input_data)
            if 'ITRANSFORMER' in selected_models:
                model_outputs['ITRANSFORMER'] = water_itransformer.forward(input_data)
            if 'TIMESNET' in selected_models:
                model_outputs['TIMESNET'] = timesnet_water.forward(input_data)

            st.write("Model Responses:")
            for model_name, output in model_outputs.items():
                if output[0] > 0.5:
                    out = "Potable"
                else:
                    out = "Unpotable"

                st.write(f"{model_name}: {out}")
        
    elif st.session_state.selected == 'Rainfall':
        st.markdown("<h3 style='text-align: center;'>Rainfall Prediction</h3>", unsafe_allow_html=True)
        selected_models = st.selectbox("Select model", ['XLSTM', 'ITRANSFORMER', 'TIMESNET'])

        if selected_models:
            if selected_models == "XLSTM":
                data = pd.read_csv(XLSTM_RAINFALL_PREDICTION)
            elif selected_models == "ITRANSFORMER":
                data = pd.read_csv(ITRANSFORMER_RAINFALL_PREDICTION)
            elif selected_models == "TIMESNET":
                data = pd.read_csv(TIMESNET_RAINFALL_PREDICTION)

            data['SUBDIVISION'] = data['SUBDIVISION'].astype('category')
            data['YEAR'] = data['YEAR'].astype('category')

            selected_subdivision = st.selectbox("Select Subdivision", data['SUBDIVISION'].unique())

            filtered_data = data[data['SUBDIVISION'] == selected_subdivision]
            years = filtered_data['YEAR'].tolist()
            filtered_data = filtered_data.iloc[:, 1:-2].T 

            filtered_data.columns = years

            if selected_subdivision:
                fig = px.line(filtered_data,
                  labels={'index': 'Month', 'value': 'Rainfall (mm)', 'variable': 'Year'},
                  title=f"Rainfall Prediction for {selected_subdivision}")

                fig.update_traces(mode='lines+markers') 
                fig.update_layout(hovermode='x unified')  

                st.plotly_chart(fig, use_container_width=True)

        with st.form(key="rainfall_form"):
            st.markdown("<h3 style='text-align: center;'>Input 5 consecutive rainfall data for prediction</h3>", unsafe_allow_html=True)

            selected_model = st.selectbox("Select model", ['XLSTM', 'ITRANSFORMER', 'TIMESNET'])
            selected_division = st.selectbox("Select Subdivision", data['SUBDIVISION'].unique())
            onehot_index = subdivision_to_idx[selected_division]

            onehot = torch.zeros((1,5,36))
            onehot[ :, :, onehot_index] = 1

            years = ['1st', '2nd', '3rd', '4th', '5th']
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            rainfall_data = []

            for year in years:
                st.markdown(f"<h4>Rainfall data for {year} year</h4>", unsafe_allow_html=True)
                cols = st.columns(4)
                data = []
                for i, month in enumerate(months):
                    col_index = i % 4
                    with cols[col_index]:
                        data.append(st.number_input(
                            f"{month}",
                            min_value=0.0,
                            format="%.2f",
                            key=f"rainfall_{year}_{month}"
                        ))
                
                rainfall_data.append(data)

            submit_button = st.form_submit_button(label="Predict Rainfall")

        if submit_button:
            rainfall_data = torch.Tensor(rainfall_data)
            rainfall_data = torch.Tensor(rainfall_scaler.transform(rainfall_data))
            inference_data = torch.cat([onehot, rainfall_data.unsqueeze(0)],dim=-1) 

            if selected_model == "XLSTM":
                out = xlstm_rainfall.forward(inference_data).detach().numpy()
            elif selected_model == "ITRANSFORMER":
                out = rainfall_itransformer.forward(inference_data).detach().numpy()
            elif selected_model == "TIMESNET":
                out = timesnet_rainfall.forward(inference_data).detach().numpy()

            out = rainfall_scaler.inverse_transform(out)
            out = torch.nn.ReLU()(torch.Tensor(out))

            fig = go.Figure(data=[
                go.Bar(name='Predicted Rainfall', x=months, y=out.numpy().flatten())
            ])

            fig.update_layout(
                title=f"Predicted Rainfall for {selected_division} using {selected_model}",
                xaxis_title="Month",
                yaxis_title="Rainfall (mm)",
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("Predicted Rainfall Values (mm):")
            df_predictions = pd.DataFrame({'Month': months, 'Predicted Rainfall (mm)': out.numpy().flatten()})
            st.table(df_predictions)

    elif st.session_state.selected == 'Weather':
        st.markdown("<h3 style='text-align: center;'>Weather Prediction</h3>", unsafe_allow_html=True)
        selected_models = st.selectbox("Select model", ['XLSTM', 'ITRANSFORMER', 'TIMESNET'])

        if selected_models:
            if selected_models == "XLSTM":
                data = pd.read_csv(XLSTM_WEATHER_PREDICTION)
            elif selected_models == "ITRANSFORMER":
                data = pd.read_csv(ITRANSFORMER_WEATHER_PREDICTION)
            elif selected_models == "TIMESNET":
                data = pd.read_csv(TIMESNET_WEATHER_PREDICTION)

            data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
            selected_year = st.selectbox("Select Year", data['year'].unique().astype(int))
            selected_city = st.selectbox("Select City", data['country'].unique())

            filtered_data = data[(data['year'] == selected_year) & (data['country'] == selected_city)]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['min'],
                mode='lines',
                name='Minimum Temperature'
            ))

            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['avg'],
                mode='lines',
                name='Average Temperature'
            ))

            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['max'],
                mode='lines',
                name='Maximum Temperature'
            ))

            # Customize the layout
            fig.update_layout(
                title=f"Temperature Trends for {selected_city} in {selected_year}",
                xaxis_title="Date",
                yaxis_title="Temperature",
                legend_title="Temperature Type",
                hovermode="x unified"
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        with st.form(key="weather_form"):
            st.markdown("<h3 style='text-align: center;'>Input last 31 weather for prediction</h3>", unsafe_allow_html=True)

            selected_model = st.selectbox("Select model", ['XLSTM', 'ITRANSFORMER', 'TIMESNET'])
            selected_division = st.selectbox("Select City", data['country'].unique())
            date = st.date_input('Pick the start date of the input data')
            pred_len = st.number_input("How many day should the model predict",min_value=1, step=1)
            onehot_index = cities_to_idx[selected_division]

            onehot = torch.zeros((1, 31, 5))
            onehot[:, :, onehot_index] = 1

            weather_data = []
            weather_data_columns = ['Minimum', 'Average', 'Maximum']

            for day in range(0, 31, 2):  
                cols = st.columns(2)
                day_1 = f"Day {day + 1}"
                day_2 = f"Day {day + 2}" if day + 2 <= 31 else None

                with cols[0]:
                    st.markdown(f"<h6>{day_1}</h6>", unsafe_allow_html=True)
                    day_1_data = []
                    for i, column in enumerate(weather_data_columns):
                        day_1_data.append(st.number_input(
                            f"{column} ({day_1})",
                            min_value=0.0,
                            format="%.2f",
                            key=f"weather_{column}_{day + 1}"
                        ))
                    weather_data.append(day_1_data)

                if day_2:  
                    with cols[1]:
                        st.markdown(f"<h6>{day_2}</h6>", unsafe_allow_html=True)
                        day_2_data = []
                        for i, column in enumerate(weather_data_columns):
                            day_2_data.append(st.number_input(
                                f"{column} ({day_2})",
                                min_value=0.0,
                                format="%.2f",
                                key=f"weather_{column}_{day + 2}"
                            ))
                        weather_data.append(day_2_data)

            submit_button = st.form_submit_button(label="Predict Weather")

        if submit_button:
            inference_data = torch.cat([onehot,torch.Tensor(weather_data).unsqueeze(0)], dim=-1)
            dates = torch.zeros((1,31,3))

            for i in range(31):
                current_date = date + timedelta(days=i)
                dates[:, i, :] = torch.tensor([current_date.year, current_date.month, current_date.day])

            inference_data = torch.cat([inference_data,dates], dim=-1)

            if 'XLSTM' in selected_models:
                out = xlstm_weather.forward(inference_data)
            if 'ITRANSFORMER' in selected_models:
                out = weather_itransformer.forward(inference_data)
            if 'TIMESNET' in selected_models:
                out = timesnet_weather.forward(inference_data)

            if pred_len == 1:
                output = [out]
            else:
                output = []
                output.append(out.detach())

                for i in range(1, pred_len):
                    onehot = torch.zeros((1, 1, 5))
                    onehot[:, :, onehot_index] = 1

                    inference_ = torch.cat([onehot, out.unsqueeze(0)], dim=-1)
                    current_date = current_date + timedelta(days=i)

                    dates = torch.zeros((1, 1, 3))
                    dates[:, :, 0] = current_date.year
                    dates[:, :, 1] = current_date.month
                    dates[:, :, 2] = current_date.day

                    inference_ = torch.cat([inference_, dates], dim=-1)

                    inference_data = torch.cat([inference_data[:, 1:, :], inference_], dim=1)

                    if 'XLSTM' in selected_models:
                        out = xlstm_weather.forward(inference_data).detach()
                    if 'ITRANSFORMER' in selected_models:
                        out = weather_itransformer.forward(inference_data).detach()
                    if 'TIMESNET' in selected_models:
                        out = timesnet_weather.forward(inference_data).detach()

                    output.append(out)

            current_date = date + timedelta(days=31)
            min_temps = []
            avg_temps = []
            max_temps = []
            dates = []

            for i, tensor in enumerate(output):
                min_temp, avg_temp, max_temp = tensor[0]
                
                min_temps.append(min_temp.item())
                avg_temps.append(avg_temp.item())
                max_temps.append(max_temp.item())

                dates.append(current_date)

                current_date = current_date + timedelta(days=1)
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=dates, y=min_temps, mode='lines+markers', name='Min Temp'))
            fig.add_trace(go.Scatter(x=dates, y=avg_temps, mode='lines+markers', name='Avg Temp'))
            fig.add_trace(go.Scatter(x=dates, y=max_temps, mode='lines+markers', name='Max Temp'))

            fig.update_layout(
                title='Temperature Over Time',
                xaxis_title='Date',
                yaxis_title='Temperature',
                xaxis=dict(tickformat="%Y-%m-%d"),
                template="plotly_dark"
            )

            st.plotly_chart(fig)

    if st.button("Back"):
        st.session_state.selected = None
        st.rerun()


st.markdown(
    """
    <style>
    button {
        width: 100%;
        height: 100%;
        font-size: 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

