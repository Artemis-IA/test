# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Définition des modèles Pydantic pour les requêtes
class Credit(BaseModel):
    Gender: int
    Age: int
    PhysicalActivityLevel: int
    HearthRate: int
    DailySteps: int
    BloodPressure_high: float
    BloodPressure_low: float

class Credit2(BaseModel):
    PhysicalActivityLevel: int
    HearthRate: int
    DailySteps: int

# Initialisation de l'application FastAPI
app = FastAPI(title="API de prédiction",
              description="API pour prédire sur la base de différents modèles",
              version="1.0.0")

# Chargement des modèles pré-entraînés
with open('model_1.pkl', 'rb') as file:
    model1 = pickle.load(file)

with open('model_2.pkl', 'rb') as file:
    model2 = pickle.load(file)

# Endpoints get
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction"}

@app.get("/info_model1", tags=["Info Model 1"])
def info_model1():
    return {
        "description": "Format de données attendu par le modèle 1",
        "fields": {
            "Gender": "int (0 pour féminin, 1 pour masculin)",
            "Age": "int (âge en années)",
            "Physical Activity Level": "int (niveau d'activité physique, par exemple 0 pour sédentaire, 1 pour légèrement actif, etc.)",
            "Heart Rate": "int (rythme cardiaque en battements par minute)",
            "Daily Steps": "int (nombre de pas quotidiens)",
            "BloodPressure_high": "float (pression artérielle systolique en mmHg)",
            "BloodPressure_low": "float (pression artérielle diastolique en mmHg)"
        }
    }

@app.get("/info_model2", tags=["Info Model 2"])
def info_model2():
    return {
        "description": "Format de données attendu par le modèle 2",
        "fields": {
            "Physical Activity Level": "int (niveau d'activité physique, par exemple 0 pour sédentaire, 1 pour légèrement actif, etc.)",
            "Heart Rate": "int (rythme cardiaque en battements par minute)",
            "Daily Steps": "int (nombre de pas quotidiens)",
            "Sleep Disorder": "int (0 pour aucun trouble du sommeil, 1 pour présence de trouble)"
        }
    }

# Endpoint pour la prédiction avec le modèle 1
@app.post("/predict1", tags=["Predict V1"])
def predict1_endpoint(credit: Credit):
    data_df = pd.DataFrame([{
        "Gender": credit.Gender,
        "Age": credit.Age,
        "Physical Activity Level": credit.PhysicalActivityLevel,
        "Heart Rate": credit.HearthRate,
        "Daily Steps": credit.DailySteps,
        "BloodPressure_high": credit.BloodPressure_high,
        "BloodPressure_low": credit.BloodPressure_low
    }])
    prediction = model1.predict(data_df)
    prediction = prediction.tolist()
    return {"prediction": prediction}


# Endpoint pour la prédiction avec le modèle 2
@app.post("/predict2", tags=["Predict V2"])
def predict2_endpoint(credit: Credit2):
    data = np.array([[credit.PhysicalActivityLevel, credit.HearthRate, credit.DailySteps]])
    prediction = model2.predict(data)
    prediction = prediction.tolist()  
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)