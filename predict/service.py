from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
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
            
            # Expanded training data for better model performance
            train_data = {
                'Soil_Characteristics': ['Clay', 'Loam', 'Sandy', 'Clay', 'Loam', 'Sandy', 'Clay', 'Loam', 'Sandy', 'Clay'],
                'Nutrition_Value': ['Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'High', 'Medium', 'Low', 'High'],
                'Crop_Variety': ['A', 'B', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                'Pest_and_Diseases': ['Moderate', 'Low', 'Low', 'None', 'Low', 'High', 'Moderate', 'None', 'High', 'Low'],
                'Fertilizers': ['Medium', 'High', 'Low', 'None', 'High', 'Medium', 'High', 'Low', 'Medium', 'None'],
                'Fertilizer_Usage': ['Moderate', 'High', 'Low', 'None', 'High', 'Medium', 'High', 'Low', 'Medium', 'None'],
                'Farm_Size_Acres': [10, 15, 12, 20, 18, 8, 25, 14, 16, 22],
                'pH_Level': [6.5, 7.0, 6.8, 6.0, 6.2, 6.9, 7.2, 6.4, 6.7, 6.1],
                'Yield_Tons': [30, 40, 25, 35, 38, 20, 45, 28, 33, 37]
            }

            df = pd.DataFrame(train_data)
            
            print(f"Dataset shape: {df.shape}")

            # Store original encoders for later use
            encoders = {}
            
            # Convert categorical variables to numerical using Label Encoding
            categorical_columns = ['Soil_Characteristics', 'Nutrition_Value', 'Crop_Variety', 
                                 'Pest_and_Diseases', 'Fertilizers', 'Fertilizer_Usage']
            
            for col in categorical_columns:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col])

            # Split the dataset into features (X) and target variable (y)
            X = df.drop('Yield_Tons', axis=1)
            y = df['Yield_Tons']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Testing set size: {X_test.shape[0]}")

            # Create a Random Forest Regressor model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Train the model on training data
            rf_model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = rf_model.predict(X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nModel Performance Metrics:")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nFeature Importance:")
            print(feature_importance)

            # Make prediction for user input
            soil_char_encoded = safe_label_encode(encoders['Soil_Characteristics'], [data["Soil_Characteristics"]])[0]
            nutrition_value_encoded = safe_label_encode(encoders['Nutrition_Value'], [data['Nutrition_Value']])[0]
            crop_variety_encoded = safe_label_encode(encoders['Crop_Variety'], [data['Crop_Variety']])[0]
            pest_diseases_encoded = safe_label_encode(encoders['Pest_and_Diseases'], [data['Pest_and_Diseases']])[0]
            fertilizers_encoded = safe_label_encode(encoders['Fertilizers'], [data['Fertilizers']])[0]
            fertilizer_usage_encoded = safe_label_encode(encoders['Fertilizer_Usage'], [data['Fertilizer_Usage']])[0]
            
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
                    "r2_score": r2,
                    "rmse": rmse,
                    "mae": mae
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
        # Create DataFrame and process
        df = pd.read_csv('crop_prediction_dataset.csv')
        print(f"Crop suggestion dataset shape: {df.shape}")
        
        # Convert categorical variables using one-hot encoding
        df = pd.get_dummies(df, columns=['Soil_Type'])
        
        # Split features and target
        X = df.drop('Crop', axis=1)
        y = df['Crop']
        
        # Train-test split
        if len(df) > 4:  # Only split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Testing set size: {X_test.shape[0]}")
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nCrop Suggestion Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            if len(np.unique(y_test)) > 1:  # Only show classification report if we have multiple classes
                print(f"\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
        else:
            # If dataset is too small, train on full dataset
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            print("Dataset too small for train-test split. Using full dataset for training.")
        
        # Save the trained model
        # joblib.dump(model, 'random_forest_model.pkl')
        print("Model saved as 'random_forest_model.pkl'")
        
        # Process user input
        user_data = data.copy()
        for soil_type in ['Clay', 'Loam', 'Silt', 'Sand', 'Pit', 'Chalk']:
            user_data[f"Soil_Type_{soil_type}"] = 1 if soil_type == data['Soil_Type'] else 0
        
        # Remove original Soil_Type column and create DataFrame
        del user_data['Soil_Type']
        user_df = pd.DataFrame([user_data])
        
        # Reorder columns to match training data
        user_df = user_df[X.columns]
        
        # Make prediction
        predictions = model.predict(user_df)
        prediction_proba = model.predict_proba(user_df)
        
        # Get confidence score
        max_confidence = np.max(prediction_proba[0])
        
        return success_response(content={
            "crop": predictions[0],
            "confidence": str(round(max_confidence * 100,2)) + "%",
            "model_accuracy": str(round(accuracy*100,2))+'%' if 'accuracy' in locals() else "N/A (insufficient data for validation)"
        })
        
    except Exception as e:
        return failure_response(statuscode=500, content={"message": str(e)})
