# 🌾 HarvestHub

![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask)
![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/Deployed%20on-AWS%20EC2-FF9900?logo=amazon-aws&logoColor=white)
![Mobile](https://img.shields.io/badge/Mobile%20Client-React%20Native-61DAFB?logo=react)
![Web](https://img.shields.io/badge/Web%20Client-React%20JS-61DAFB?logo=react)

---

## 🚀 Project Overview

HarvestHub is an ML-powered crop recommendation and yield prediction system. It helps farmers decide **what to plant** and **estimate crop yield** using soil and environmental data. The backend is a Flask-based API, containerized with Docker, and deployed to AWS EC2. Clients can interact with the API from any HTTP-capable environment like mobile apps, web clients, or scripts.

---

## 🧠 Machine Learning Components

- **Model Types**:
  - `RandomForestClassifier` for crop recommendation
  - `RandomForestRegressor` for yield prediction

- **Dataset (Sample Schema)**:
    ```json
    {
      "Soil_Type": ["Clay", "Loam", "Silt", "Sand", "Pit", "Chalk"],
      "Nutrition_Value": [30, 40, 25, 35, 45, 30],
      "Temperature": [25, 28, 20, 30, 22, 28],
      "Humidity": [60, 70, 50, 80, 45, 75],
      "Crop": ["Wheat", "Maize", "Rice", "Barley", "Corn", "Soybean"]
    }
    ```

- **Features**:
  - Crop Suggestion: `Soil_Type`, `Nutrition_Value`, `Temperature`, `Humidity`
  - Yield Prediction: Adds `Fertilizers`, `Farm Size`, `PH level`, etc.

- **Crop Prediction Model Accuracy**: **72.00%**
- **Yield Prediction Metrics**:
  - MAE: `6.7`
  - RMSE: `7.13`
  - R² Score: `0.26`

---

## 📦 Project Structure

```
HarvestHub/
├── app.py                     # Main Flask app
├── factory.py                 # Flask app factory
├── model.pkl                  # Classifier model
├── random_forest_model.pkl    # Regressor model
├── crop_prediction_dataset.csv
├── datasetgen.py
├── Crop_reccomendation_ipynb_json.ipynb
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── dockerpull.sh / dockerpush.sh
├── common/                   # Utilities and constants
│   ├── response.py
│   └── strings.py
├── predict/                  # Prediction logic and routes
│   ├── service.py
│   └── view.py
└── README.md
```

---

## 🔗 API Endpoints

### 🔍 Crop Suggestion

**POST** `/pred/suggest`

```bash
curl --location 'http://localhost:5000/pred/suggest' \
--header 'Content-Type: application/json' \
--data '{
    "Soil_Type":"Clay",
    "Nutrition_Value":35,
    "Temperature":26,
    "Humidity":70
}'
```

**Sample Response**:
```json
{
  "content": {
    "confidence": "65.0%",
    "crop": "Rice",
    "model_accuracy": "72.0%"
  },
  "status": "Success",
  "status_code": 200
}
```

---

### 📊 Yield Prediction

**POST** `/pred/predict`

```bash
curl --location 'http://localhost:5000/pred/predict' \
--header 'Content-Type: application/json' \
--data '{
    "Soil_Characteristics":"Clay",
    "Nutrition_Value":"High",
    "Crop_Variety":"A",
    "Pest_and_Diseases":"Moderate",
    "Fertilizers":"asd",
    "Fertilizer_Usage":"High",
    "farm_size_acres":100,
    "ph_level":6
}'
```

**Sample Response**:
```json
{
    "content": {
        "model_metrics": {
            "mae": 1.5062800000000005,
            "r2_score": "89.35%",
            "rmse": 2.449508179206594
        },
        "predicted_yield_in_tons": 21.46
    },
    "status": "Success",
    "status_code": 200
}
```

---

## 🐳 Docker Deployment

```bash
docker build -t harvesthub-api .
docker run -d -p 5000:5000 harvesthub-api
```

Or use `docker-compose`:

```bash
docker-compose up -d
```

---

## 🌐 Hosting Info

- Hosted on **AWS EC2** using Docker
- Accessible via **HTTP API** from mobile and web clients

---

**Prasanth M**  
Email: prasanthm6379@gmail.com  
GitHub: github.com/prasanth6379
LinkedIn: https://www.linkedin.com/in/prasanth-m-07397018a/

---
