from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
from common.response import failure_response, success_response
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle

def safe_label_encode(encoder, labels):
    unknown_labels = set(labels) - set(encoder.classes_)
    for label in unknown_labels:
        encoder.classes_ = np.append(encoder.classes_, label)
    return encoder.transform(labels)


def get_prediction(data):
    try:
        if len(data) == 8:

            encoders = {}
            df = pd.read_csv('yield_prediction_dataset.csv')

            categorical_columns = ['Soil_Characteristics', 'Nutrition_Value', 'Crop_Variety', 
                                 'Pest_and_Diseases', 'Fertilizers', 'Fertilizer_Usage']
            
            for col in categorical_columns:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col])

            soil_char_encoded = safe_label_encode(encoders['Soil_Characteristics'], [data["Soil_Characteristics"]])[0]
            nutrition_value_encoded = safe_label_encode(encoders['Nutrition_Value'], [data['Nutrition_Value']])[0]
            crop_variety_encoded = safe_label_encode(encoders['Crop_Variety'], [data['Crop_Variety']])[0]
            pest_diseases_encoded = safe_label_encode(encoders['Pest_and_Diseases'], [data['Pest_and_Diseases']])[0]
            fertilizers_encoded = safe_label_encode(encoders['Fertilizers'], [data['Fertilizers']])[0]
            fertilizer_usage_encoded = safe_label_encode(encoders['Fertilizer_Usage'], [data['Fertilizer_Usage']])[0]

            rf_model = pickle.load(open('yield_prediction.pkl', 'rb'))
            
            
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
            
            return success_response(statuscode=200, content={
                "predicted_yield_in_tons": val[0],
                "model_metrics": {
                    "r2_score": "0.89",
                    "rmse": "2.4495",
                    "mae": '1.5063'
                }
            })
        else:
            return failure_response(statuscode=400, content={"message": "Data count mismatch"})
    except Exception as e:
        return failure_response(statuscode=500, content={"message": str(e)})


def get_suggestion(data):
    if len(data) != 4:
        return failure_response(content={'message': "Data count mismatch"})
    
    try:       
        with open('crop_suggestion.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        soil_types = model_data['soil_types']
        
        user_data = data.copy()
        
        soil_type_input = user_data['Soil_Type']
        
        del user_data['Soil_Type']
        
        for soil_type in soil_types:
            user_data[f"Soil_Type_{soil_type}"] = 1 if soil_type == soil_type_input else 0
        
        user_df = pd.DataFrame([user_data])
        
        user_df = user_df.reindex(columns=feature_columns, fill_value=0)
        
        
        predictions = model.predict(user_df)
        prediction_proba = model.predict_proba(user_df)
        
        max_confidence = np.max(prediction_proba[0])
        
        return success_response(content={
            "crop": predictions[0],
            "confidence": str(round(max_confidence * 100, 2)) + "%",
            "model_accuracy": "72.00%"
        })
        
    except Exception as e:
        return failure_response(statuscode=500, content={"message": str(e)})
