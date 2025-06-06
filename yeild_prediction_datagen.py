


import pandas as pd
import numpy as np
import random

def generate_yield_dataset(n_samples=1000, random_seed=42):
    """
    Generate a realistic agricultural dataset for yield prediction.
    
    The dataset considers real agricultural relationships:
    - Clay soil retains nutrients better but may have drainage issues
    - Loam is ideal for most crops with balanced properties
    - Sandy soil drains well but requires more fertilizer
    - Higher nutrition and fertilizer usage generally increase yield
    - Pest and disease pressure reduces yield
    - pH affects nutrient availability
    - Farm size can affect management efficiency
    """
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = {
        'Soil_Characteristics': [],
        'Nutrition_Value': [],
        'Crop_Variety': [],
        'Pest_and_Diseases': [],
        'Fertilizers': [],
        'Fertilizer_Usage': [],
        'Farm_Size_Acres': [],
        'pH_Level': [],
        'Yield_Tons': []
    }
    
    # Define categories and their relationships
    soil_types = ['Clay', 'Loam', 'Sandy']
    nutrition_levels = ['Low', 'Medium', 'High']
    crop_varieties = ['A', 'B', 'C']
    pest_disease_levels = ['None', 'Low', 'Moderate', 'High']
    fertilizer_levels = ['None', 'Low', 'Medium', 'High']
    fertilizer_usage_levels = ['None', 'Low', 'Medium', 'Moderate', 'High']
    
    # Define base yield potential for different combinations
    soil_yield_multiplier = {
        'Clay': 0.9,      # Good nutrients but potential drainage issues
        'Loam': 1.0,      # Ideal soil type
        'Sandy': 0.85     # Good drainage but nutrients leach away
    }
    
    nutrition_yield_multiplier = {
        'Low': 0.7,
        'Medium': 0.9,
        'High': 1.1
    }
    
    crop_variety_base_yield = {
        'A': 30,  # Standard variety
        'B': 35,  # Higher yielding variety
        'C': 25   # Lower yielding but maybe more resistant
    }
    
    pest_disease_yield_multiplier = {
        'None': 1.0,
        'Low': 0.95,
        'Moderate': 0.85,
        'High': 0.7
    }
    
    fertilizer_yield_multiplier = {
        'None': 0.8,
        'Low': 0.9,
        'Medium': 1.0,
        'High': 1.15
    }
    
    fertilizer_usage_yield_multiplier = {
        'None': 0.8,
        'Low': 0.9,
        'Medium': 1.0,
        'Moderate': 1.05,
        'High': 1.1
    }
    
    for i in range(n_samples):
        # Select soil type
        soil_type = random.choice(soil_types)
        
        # Nutrition value influenced by soil type
        if soil_type == 'Clay':
            nutrition = np.random.choice(nutrition_levels, p=[0.2, 0.5, 0.3])  # Clay retains nutrients
        elif soil_type == 'Loam':
            nutrition = np.random.choice(nutrition_levels, p=[0.2, 0.4, 0.4])  # Loam is balanced
        else:  # Sandy
            nutrition = np.random.choice(nutrition_levels, p=[0.4, 0.4, 0.2])  # Sandy loses nutrients
        
        # Select crop variety
        crop_variety = random.choice(crop_varieties)
        
        # Pest and disease influenced by management and environmental factors
        pest_disease = random.choice(pest_disease_levels)
        
        # Fertilizer application
        fertilizer = random.choice(fertilizer_levels)
        
        # Fertilizer usage (can be different from fertilizer type)
        # High correlation with fertilizer but some variation
        if fertilizer == 'None':
            fertilizer_usage = np.random.choice(['None', 'Low'], p=[0.8, 0.2])
        elif fertilizer == 'Low':
            fertilizer_usage = np.random.choice(['Low', 'Medium'], p=[0.7, 0.3])
        elif fertilizer == 'Medium':
            fertilizer_usage = np.random.choice(['Medium', 'Moderate', 'High'], p=[0.5, 0.3, 0.2])
        else:  # High
            fertilizer_usage = np.random.choice(['Moderate', 'High'], p=[0.3, 0.7])
        
        # Farm size - affects management efficiency
        if random.random() < 0.3:  # 30% small farms
            farm_size = np.random.uniform(8, 15)
        elif random.random() < 0.6:  # 30% medium farms
            farm_size = np.random.uniform(15, 25)
        else:  # 40% large farms
            farm_size = np.random.uniform(25, 50)
        
        farm_size = round(farm_size, 1)
        
        # pH level influenced by soil type
        if soil_type == 'Clay':
            ph_level = np.random.normal(6.8, 0.4)  # Clay tends to be more alkaline
        elif soil_type == 'Loam':
            ph_level = np.random.normal(6.5, 0.3)  # Loam is near neutral
        else:  # Sandy
            ph_level = np.random.normal(6.2, 0.3)  # Sandy tends to be more acidic
        
        ph_level = np.clip(ph_level, 5.5, 8.0)
        ph_level = round(ph_level, 1)
        
        # Calculate yield based on all factors
        base_yield = crop_variety_base_yield[crop_variety]
        
        # Apply multipliers
        yield_tons = base_yield
        yield_tons *= soil_yield_multiplier[soil_type]
        yield_tons *= nutrition_yield_multiplier[nutrition]
        yield_tons *= pest_disease_yield_multiplier[pest_disease]
        yield_tons *= fertilizer_yield_multiplier[fertilizer]
        yield_tons *= fertilizer_usage_yield_multiplier[fertilizer_usage]
        
        # Farm size effect (larger farms may have economies of scale but also management challenges)
        if farm_size < 15:
            yield_tons *= np.random.uniform(0.95, 1.05)  # Small farms - variable management
        elif farm_size < 30:
            yield_tons *= np.random.uniform(1.0, 1.1)    # Medium farms - good management
        else:
            yield_tons *= np.random.uniform(0.98, 1.08)  # Large farms - management challenges
        
        # pH effect on yield
        optimal_ph = 6.5
        ph_deviation = abs(ph_level - optimal_ph)
        ph_multiplier = max(0.8, 1.0 - (ph_deviation * 0.1))
        yield_tons *= ph_multiplier
        
        # Add some random variation (weather, other factors)
        yield_tons *= np.random.uniform(0.9, 1.1)
        
        # Ensure realistic yield range
        yield_tons = np.clip(yield_tons, 15, 60)
        yield_tons = round(yield_tons)
        
        # Add some correlation adjustments for realism
        if random.random() < 0.05:  # 5% exceptional cases
            if fertilizer == 'High' and fertilizer_usage == 'High' and pest_disease == 'None':
                yield_tons = min(60, yield_tons * 1.2)  # Exceptional conditions
            elif fertilizer == 'None' and pest_disease == 'High':
                yield_tons = max(15, yield_tons * 0.7)  # Poor conditions
        
        # Store data
        data['Soil_Characteristics'].append(soil_type)
        data['Nutrition_Value'].append(nutrition)
        data['Crop_Variety'].append(crop_variety)
        data['Pest_and_Diseases'].append(pest_disease)
        data['Fertilizers'].append(fertilizer)
        data['Fertilizer_Usage'].append(fertilizer_usage)
        data['Farm_Size_Acres'].append(farm_size)
        data['pH_Level'].append(ph_level)
        data['Yield_Tons'].append(yield_tons)
    
    return pd.DataFrame(data)

# Generate the dataset
if __name__ == "__main__":
    # Generate dataset
    df = generate_yield_dataset(n_samples=1000, random_seed=42)
    
    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nFirst 15 rows:")
    print(df.head(15))
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nCategorical Variables Distribution:")
    categorical_cols = ['Soil_Characteristics', 'Nutrition_Value', 'Crop_Variety', 
                       'Pest_and_Diseases', 'Fertilizers', 'Fertilizer_Usage']
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    print("\nNumerical Variables Summary:")
    numerical_cols = ['Farm_Size_Acres', 'pH_Level', 'Yield_Tons']
    print(df[numerical_cols].describe())
    
    # Analyze relationships
    print("\n" + "="*50)
    print("RELATIONSHIP ANALYSIS")
    print("="*50)
    
    # Average yield by soil type
    print("\nAverage Yield by Soil Type:")
    print(df.groupby('Soil_Characteristics')['Yield_Tons'].mean().sort_values(ascending=False))
    
    # Average yield by nutrition level
    print("\nAverage Yield by Nutrition Level:")
    print(df.groupby('Nutrition_Value')['Yield_Tons'].mean().sort_values(ascending=False))
    
    # Average yield by crop variety
    print("\nAverage Yield by Crop Variety:")
    print(df.groupby('Crop_Variety')['Yield_Tons'].mean().sort_values(ascending=False))
    
    # Average yield by pest/disease level
    print("\nAverage Yield by Pest/Disease Level:")
    print(df.groupby('Pest_and_Diseases')['Yield_Tons'].mean().sort_values(ascending=False))
    
    # Save to CSV
    df.to_csv('yield_prediction_dataset.csv', index=False)