import os

import joblib
from sklearn.ensemble import RandomForestRegressor

from common.response import failure_response, success_response
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import warnings


def safe_label_encode(encoder, labels):
    unknown_labels = set(labels) - set(encoder.classes_)
    for label in unknown_labels:
        encoder.classes_ = np.append(encoder.classes_, label)
    return encoder.transform(labels)


def get_prediction(data):
    try:
        if len(data) == 8:
            warnings.filterwarnings('ignore')
            train_data = {
                'Soil_Characteristics': ['Clay', 'Loam', 'Sandy', 'Clay', 'Loam'],
                'Nutrition_Value': ['Medium', 'High', 'Low', 'Medium', 'High'],
                'Crop_Variety': ['A', 'B', 'A', 'B', 'A'],
                'Pest_and_Diseases': ['Moderate', 'Low', 'Low', 'None', 'Low'],
                'Fertilizers': ['Medium', 'High', 'Low', 'None', 'High'],
                'Fertilizer_Usage': ['Moderate', 'High', 'Low', 'None', 'High'],
                'Farm_Size_Acres': [10, 15, 12, 20, 18],
                'pH_Level': [6.5, 7.0, 6.8, 6.0, 6.2],
                'Yield_Tons': [30, 40, 25, 35, 38]
            }

            df = pd.DataFrame(train_data)

            # Convert categorical variables to numerical using Label Encoding
            label_encoder = LabelEncoder()
            df['Soil_Characteristics'] = label_encoder.fit_transform(df['Soil_Characteristics'])
            df['Nutrition_Value'] = label_encoder.fit_transform(
                df['Nutrition_Value'])  # Encoding 'Medium','High', 'Low'
            df['Crop_Variety'] = label_encoder.fit_transform(df['Crop_Variety'])
            df['Pest_and_Diseases'] = label_encoder.fit_transform(df['Pest_and_Diseases'])
            df['Fertilizers'] = label_encoder.fit_transform(df['Fertilizers'])
            df['Fertilizer_Usage'] = label_encoder.fit_transform(df['Fertilizer_Usage'])

            # Split the dataset into features (X) and target variable (y)
            X = df.drop('Yield_Tons', axis=1)
            y = df['Yield_Tons']

            # Create a Random Forest Regressor model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Train the model
            rf_model.fit(X, y)
            # label_encoder.fit_transform([data['Soil_Characteristics']])
            soil_char_encoded = safe_label_encode(label_encoder, [data["Soil_Characteristics"]])[0]
            nutrition_value_encoded = safe_label_encode(label_encoder, [data['Nutrition_Value']])[0]
            crop_variety_encoded = safe_label_encode(label_encoder, [data['Crop_Variety']])[0]
            pest_diseases_encoded = safe_label_encode(label_encoder, [data['Pest_and_Diseases']])[0]
            fertilizers_encoded = safe_label_encode(label_encoder, [data['Fertilizers']])[0]
            fertilizer_usage_encoded = safe_label_encode(label_encoder, [data['Fertilizer_Usage']])[0]
            # print(data)
            val = rf_model.predict([[
                soil_char_encoded,
                nutrition_value_encoded,
                crop_variety_encoded,
                pest_diseases_encoded,
                fertilizers_encoded,
                fertilizer_usage_encoded,
                data["farm_size_acres"],
                data["ph_level"]
            ]])
            return success_response(statuscode=200, content={"predicted_yield": val[0]})
        else:
            return failure_response(statuscode=400, content={"message": "Data count mismatch"})
    except Exception as e:
        return failure_response(statuscode=500, content={"message": str(e)})


def get_suggestion(data):
    if len(data) != 4:
        return failure_response(content={'message': "Data count mismatch"})
    try:
        for soil_type in ['Clay', 'Loam', 'Silt', 'Sand', 'Pit', 'Chalk']:
            data[f"Soil_Type_{soil_type}"] = 1 if soil_type == data['Soil_Type'] else 0
        data = pd.DataFrame([data])
        data = data[['Nutrition_Value', 'Temperature', 'Humidity', 'Soil_Type_Chalk',
                     'Soil_Type_Clay', 'Soil_Type_Loam', 'Soil_Type_Pit', 'Soil_Type_Sand',
                     'Soil_Type_Silt']]
        model = joblib.load('random_forest_model.pkl')

        predictions = model.predict(data)
        return success_response(content={"crop": predictions[0]})
    except Exception as e:
        return failure_response(statuscode=500, content={"message": str(e)})