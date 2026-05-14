from fastapi import FastAPI, Body
from typing import Union, List, Dict
from typing import Union, List, Dict, Any


from src.predict import make_prediction 

app = FastAPI()

@app.get("/")
def root():
    return {"Hello, welcome to Customer Churn by BiQuant inc."}

@app.post("/predict")
def predict(data: Union[List[Any], Dict[str, Any]] = Body(...)):
    try:
        if isinstance(data, dict) and "data" in data:
            payload = data["data"]
        else:
            payload = data
        prediction = make_prediction(payload)
        return {"prediction": prediction}
    except Exception as e:
        return {"status": "error", "message": str(e)}
