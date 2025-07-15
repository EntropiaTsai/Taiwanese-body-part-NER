from fastapi import FastAPI, HTTPException
from schemas import InputText, BatchInput, PredictionOutput, BatchOutput
from predictor import run_prediction
from text_generation import call_gemini, gereration, extract_json_block
import json

app = FastAPI()

@app.post("/predict_batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    results = [run_prediction(text) for text in batch.texts]
    gemini_output = gereration(results)
    print(gemini_output)
    parsed = extract_json_block(gemini_output)

    if parsed is None or "results" not in parsed:
        raise HTTPException(status_code=500, detail="Gemini 回傳格式錯誤或解析失敗")
    
    return parsed