from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = FastAPI()

MODEL_PATH = "kailermai03/techJamPII" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

class InputText(BaseModel):
    text: str

def clean_predictions(predictions):
    results = []
    current_entity = None
    current_word = ""

    for pred in predictions:
        score = float(pred["score"])
        entity = pred["entity_group"]
        word = pred["word"]

        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_entity:
                results.append({
                    "entity_group": current_entity,
                    "word": current_word,
                    "score": score
                })
            current_entity = entity
            current_word = word

    if current_entity:
        results.append({
            "entity_group": current_entity,
            "word": current_word,
            "score": score
        })
    return results

@app.post("/predict")
async def predict(input: InputText):
    results = nlp(input.text)
    cleaned = clean_predictions(results)
    return {"entities": cleaned}

@app.get("/")
async def home():
    return {"message": "NER API is running"}
