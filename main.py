
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RUN_ID ="f722fec74b9f4b9dae3f04e06acb873c"	

model = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

app = FastAPI(title="Employee Attrition Prediction API")


class EmployeeInput(BaseModel):
    Age: int
    Department: str
    JobRole: str
    MonthlyIncome: float
    OverTime: str
    JobSatisfaction: int


@app.get("/")
def home():
    return {"message": "Employee Attrition API is running"}


@app.post("/predict")
def predict(data: EmployeeInput):
    try:
        employee = pd.DataFrame([data.model_dump()])

        prediction = int(model.predict(employee)[0])
        probability = float(model.predict_proba(employee)[0, 1])

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "attrition_probability": round(probability, 4)
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
