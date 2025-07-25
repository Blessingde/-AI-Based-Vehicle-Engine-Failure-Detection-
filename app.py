# import libraries
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Title
st.write("🚗 Engine Fault Prediction Dashboard")

# user_file_upload
upload_file = st.file_uploader("📁 Upload your Vehicle Sensor CSV file/Obd (on-board diagnostics)", type=['csv'])

# Required features
FEATURE_COLUMNS =  ["Temperature (°C)", "RPM", "Fuel_Efficiency", "Vibration_X",
                    "Vibration_Y", "Vibration_Z", "Torque", "Power_Output (kW)",
                    "Operational_Mode",
                    ]

if st.button('Get analysis'):
    if upload_file is not None:
        # Check for CSV
        if upload_file.name.endswith('.csv'):
            try:
                # Load data
                df = pd.read_csv(upload_file)
                input_features = df[FEATURE_COLUMNS]
                st.write('File Uploaded Successfully')
                st.write(input_features.head())

                #  Load model
                encoder = joblib.load('./model/encoder.pkl')
                model = joblib.load("./model/model.pkl")

                # Encode the categorical column in the uploaded csv
                input_features_encoded = encoder.transform(input_features)

                # Model prediction
                predictions = model.predict(input_features_encoded)
                df['Predicted_Fault'] = predictions
                st.subheader("Predicted Fault by the model")
                st.dataframe(df.head())

                # --- Rolling average logic ---
                df['Fault_Rolling'] = df['Predicted_Fault'].rolling(window=40).mean()
                df['Health_Status'] = df['Fault_Rolling'].apply(lambda x: 'Healthy' if x > 0.5 else 'Faulty')
                df['Health_Status_Binary'] = df['Health_Status'].map({'Healthy': 1, 'Faulty': 0})

                # --- Rolling trend Visualization
                st.subheader("Rolling Prediction(Engine Health trend)")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df['Time_Stamp'], df['Fault_Rolling'], color='blue')
                # Add title and labels
                ax.set_title('Fault Rolling Over Time')
                ax.set_xlabel('Time Stamp')
                ax.set_ylabel('Fault Rolling Average')
                st.pyplot(fig)

                # Display summary
                faulty_percent = (df['Health_Status_Binary'].value_counts(normalize=True)[0] * 100)
                healthy_percent = 100 - faulty_percent

                st.subheader("Vehicle Health Summary")
                st.write(f" Faulty Prediction: **{faulty_percent:.2f}%**")
                st.write(f" Healthy Prediction: **{healthy_percent:.2f}%**")

                # Pie chart
                labels = ['Faulty', 'Healthy']
                sizes = [faulty_percent, healthy_percent]
                colors = ['#E53935', '#43A047']

                fig, ax = plt.subplots(figsize=(6,4))
                fig.patch.set_facecolor('#ffffff')  # Set figure background color
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                st.pyplot(fig)

                # Advice
                st.header('System Recommendation')
                if faulty_percent > 40:
                    st.write(f'Good: {faulty_percent:.2f} of readings show engine is faulty')
                    st.warning("⚠️ Your engine shows signs of possible faults. Please schedule a maintenance check.")
                else:
                    st.write(f'Good: {healthy_percent:.2f} of readings show engine is healthy')
                    st.success("✅ Suggestion: Continue regular maintenance and Keep monitoring regularly.")

            except Exception as e:
                st.error(f"An error occurred while processing the CSV file: {e}")
        else:
            st.error("❌ Invalid file format. Please upload a CSV file.")
    else:
        st.info("📌 Please upload a CSV file to get started.")

