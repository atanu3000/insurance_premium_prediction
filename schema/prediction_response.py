from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    prediction_category: str = Field(..., description="Predicted category of the health insurance premium", example="High")
    confidence: float = Field(..., description="Confidence level of the prediction", example=0.85)
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each class", example={"Low": 0.1, "Medium": 0.15, "High": 0.75})
