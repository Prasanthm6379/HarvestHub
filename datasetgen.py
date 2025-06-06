
import pandas as pd
import numpy as np
import random

def generate_crop_dataset(n_samples=1000, random_seed=42):
    """
    Generate a realistic agricultural dataset for crop prediction.
    
    The dataset considers real agricultural relationships:
    - Rice prefers clay soil, high humidity, moderate temperature
    - Wheat grows well in loam, moderate conditions
    - Maize/Corn prefer well-drained soils, warm temperatures
    - Barley is adaptable but prefers cooler conditions
    - Soybean needs good drainage, moderate nutrition
    """
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = {
        'Soil_Type': [],
        'Nutrition_Value': [],
        'Temperature': [],
        'Humidity': [],
        'Crop': []
    }
    
    # Define crop preferences based on agricultural knowledge
    crop_profiles = {
        'Rice': {
            'preferred_soils': ['Clay', 'Silt'],
            'soil_weights': [0.6, 0.4],
            'temp_range': (22, 32),
            'humidity_range': (70, 95),
            'nutrition_range': (25, 45)
        },
        'Wheat': {
            'preferred_soils': ['Loam', 'Clay', 'Silt'],
            'soil_weights': [0.5, 0.3, 0.2],
            'temp_range': (15, 25),
            'humidity_range': (40, 70),
            'nutrition_range': (30, 50)
        },
        'Maize': {
            'preferred_soils': ['Loam', 'Sand', 'Silt'],
            'soil_weights': [0.5, 0.3, 0.2],
            'temp_range': (20, 30),
            'humidity_range': (50, 80),
            'nutrition_range': (35, 55)
        },
        'Corn': {  # Similar to Maize but slightly different preferences
            'preferred_soils': ['Loam', 'Sand'],
            'soil_weights': [0.6, 0.4],
            'temp_range': (18, 28),
            'humidity_range': (45, 75),
            'nutrition_range': (30, 50)
        },
        'Barley': {
            'preferred_soils': ['Loam', 'Clay', 'Chalk'],
            'soil_weights': [0.4, 0.3, 0.3],
            'temp_range': (12, 22),
            'humidity_range': (35, 65),
            'nutrition_range': (25, 45)
        },
        'Soybean': {
            'preferred_soils': ['Loam', 'Silt', 'Sand'],
            'soil_weights': [0.5, 0.3, 0.2],
            'temp_range': (20, 30),
            'humidity_range': (50, 75),
            'nutrition_range': (30, 50)
        }
    }
    
    # Generate samples for each crop
    crops = list(crop_profiles.keys())
    samples_per_crop = n_samples // len(crops)
    
    for crop in crops:
        profile = crop_profiles[crop]
        
        for _ in range(samples_per_crop):
            # Select soil type based on crop preferences
            soil_type = np.random.choice(
                profile['preferred_soils'], 
                p=profile['soil_weights']
            )
            
            # Generate temperature with some variation
            temp_min, temp_max = profile['temp_range']
            temperature = np.random.normal(
                (temp_min + temp_max) / 2, 
                (temp_max - temp_min) / 6
            )
            temperature = np.clip(temperature, temp_min - 5, temp_max + 5)
            
            # Generate humidity with some variation
            hum_min, hum_max = profile['humidity_range']
            humidity = np.random.normal(
                (hum_min + hum_max) / 2, 
                (hum_max - hum_min) / 6
            )
            humidity = np.clip(humidity, hum_min - 10, hum_max + 10)
            
            # Generate nutrition value with soil type influence
            nut_min, nut_max = profile['nutrition_range']
            base_nutrition = np.random.uniform(nut_min, nut_max)
            
            # Soil type modifiers for nutrition
            soil_nutrition_modifier = {
                'Clay': 1.1,      # Clay retains nutrients well
                'Loam': 1.2,      # Loam is ideal for most crops
                'Silt': 1.0,      # Silt is moderate
                'Sand': 0.8,      # Sand drains nutrients quickly
                'Pit': 0.9,       # Pit soil is variable
                'Chalk': 0.85     # Chalk can be nutrient-poor
            }
            
            nutrition = base_nutrition * soil_nutrition_modifier.get(soil_type, 1.0)
            nutrition = np.clip(nutrition, 15, 70)
            
            # Add some realistic noise and edge cases
            if random.random() < 0.1:  # 10% of samples have some variation
                temperature += np.random.normal(0, 3)
                humidity += np.random.normal(0, 5)
                nutrition += np.random.normal(0, 5)
            
            data['Soil_Type'].append(soil_type)
            data['Nutrition_Value'].append(round(nutrition, 1))
            data['Temperature'].append(round(temperature, 1))
            data['Humidity'].append(round(humidity, 1))
            data['Crop'].append(crop)
    
    # Fill remaining samples if n_samples is not evenly divisible
    remaining_samples = n_samples - len(data['Crop'])
    for _ in range(remaining_samples):
        crop = random.choice(crops)
        profile = crop_profiles[crop]
        
        soil_type = np.random.choice(
            profile['preferred_soils'], 
            p=profile['soil_weights']
        )
        
        temp_min, temp_max = profile['temp_range']
        temperature = np.random.uniform(temp_min, temp_max)
        
        hum_min, hum_max = profile['humidity_range']
        humidity = np.random.uniform(hum_min, hum_max)
        
        nut_min, nut_max = profile['nutrition_range']
        nutrition = np.random.uniform(nut_min, nut_max)
        
        data['Soil_Type'].append(soil_type)
        data['Nutrition_Value'].append(round(nutrition, 1))
        data['Temperature'].append(round(temperature, 1))
        data['Humidity'].append(round(humidity, 1))
        data['Crop'].append(crop)
    
    # Shuffle the data
    indices = list(range(len(data['Crop'])))
    random.shuffle(indices)
    
    for key in data:
        data[key] = [data[key][i] for i in indices]
    
    return pd.DataFrame(data)

# Generate the dataset
if __name__ == "__main__":
    # Generate dataset
    df = generate_crop_dataset(n_samples=1000, random_seed=42)
    
    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nCrop Distribution:")
    print(df['Crop'].value_counts())
    
    print("\nSoil Type Distribution:")
    print(df['Soil_Type'].value_counts())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv('crop_prediction_dataset.csv', index=False)
    print("\nDataset saved as 'crop_prediction_dataset.csv'")