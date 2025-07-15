# schemas.py

from pydantic import BaseModel, Field
from typing import List

class InputText(BaseModel):
    text: str = Field(..., example="伊的面皮足薄。")

class BatchInput(BaseModel):
    texts: List[str] = Field(
        ..., 
        example=["伊的面皮足薄。", "伊的面皮足白。"]
    )

class PredictionOutput(BaseModel):
    tokens: List[str]
    labels: List[str]

class BatchOutput(BaseModel):
    results: List[PredictionOutput]
