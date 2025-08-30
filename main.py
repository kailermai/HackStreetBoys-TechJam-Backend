from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Define FastAPI app
app = FastAPI()

# 2. Load your trained model + tokenizer
MODEL_PATH = "kailermai03/techJamPII" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. Define input schema
class InputText(BaseModel):
    text: str

def clean_predictions(predictions):
    results = []
    current_entity = None
    current_word = ""

    for pred in predictions:
        # Convert numpy.float32 to native float
        score = float(pred["score"])
        entity = pred["entity_group"]
        word = pred["word"]

        # Handle subword tokens
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_entity:  # save previous entity
                results.append({
                    "entity_group": current_entity,
                    "word": current_word,
                    "score": score
                })
            current_entity = entity
            current_word = word

    # append last entity
    if current_entity:
        results.append({
            "entity_group": current_entity,
            "word": current_word,
            "score": score
        })
    return results

# 4. Define prediction endpoint
@app.post("/predict")
async def predict(input: InputText):
    results = nlp(input.text)
    cleaned = clean_predictions(results)
    return {"entities": cleaned}

# 5. Root endpoint (optional)
@app.get("/")
async def home():
    return {"message": "NER API is running"}
