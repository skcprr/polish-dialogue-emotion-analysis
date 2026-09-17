from contextlib import asynccontextmanager
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models", "best_aspectemo_model"))
TOKENIZER_NAME = "allegro/herbert-base-cased"

EMOTION_MODEL = "visegradmedia-emotion/Emotion_RoBERTa_polish6"

tokenizer = None
model = None

emotion_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, emotion_pipeline
    
    print("Loading model AspectEmo...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File not found {MODEL_PATH}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("Model ABSA loaded")
    except Exception as e:
        print(f"Error during loading ABSA model: {e}")
        raise e
        
    print(f"Loading emotion model ({EMOTION_MODEL})")
    try:
        device = 0 if torch.cuda.is_available() else -1
        emotion_pipeline = pipeline(
            "text-classification", 
            model=EMOTION_MODEL, 
            device=device
        )
        print("Emotion model loaded")
    except Exception as e:
        print(f"Error in loading emotion model: {e}")
        raise e
        
    yield
    
    print("Closing app")
    del model
    del tokenizer
    del emotion_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Polska analiza aspektów i emocji API (Projekt 15)",
    description="Zintegrowany system ABSA oraz klasyfikacji emocji w polskich tekstach obsługi klienta.",
    version="4.0.0",
    lifespan=lifespan
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_all(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    if any(v is None for v in [model, tokenizer, emotion_pipeline]):
        raise HTTPException(status_code=503, detail="Models aren't loaded yet")
    

    try:
        emotion_res = emotion_pipeline(request.text)[0]
        raw_label = emotion_res["label"].lower()
        emo_confidence = round(emotion_res["score"] * 100, 1)
        
        label_map = {
            "anger": "złość",
            "fear": "strach",
            "disgust": "wstręt",
            "sadness": "smutek",
            "joy": "radość",
            "none of them": "neutralny",
            "none": "neutralny",
            "label_0": "złość",
            "label_1": "strach",
            "label_2": "wstręt",
            "label_3": "smutek",
            "label_4": "radość",
            "label_5": "neutralny"
        }
        polish_emotion = label_map.get(raw_label, raw_label)
    except Exception as e:
        print(f"Błąd potoku emocji: {e}")
        polish_emotion = "nieznany"
        emo_confidence = 0.0


    inputs = tokenizer(request.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    clean_words = []
    current_word = ""
    current_labels = []
    current_confidences = []
    
    for token, pred_id, token_probs in zip(tokens, predictions, probabilities):
        if token in ["<s>", "</s>", "<pad>", "<unk>"]:
            continue
            
        label = model.config.id2label[pred_id]
        confidence = float(token_probs[pred_id])
        
        if token.endswith("</w>"):
            current_word += token.replace("</w>", "")
            current_labels.append(label)
            current_confidences.append(confidence)
            
            final_label = max(set(current_labels), key=current_labels.count)
            avg_confidence = round((sum(current_confidences) / len(current_confidences)) * 100, 1)
            
            clean_words.append({
                "word": current_word,
                "predicted_label": final_label,
                "confidence": f"{avg_confidence}%"
            })
            
            current_word = ""
            current_labels = []
            current_confidences = []
        else:
            current_word += token
            current_labels.append(label)
            current_confidences.append(confidence)

    if current_word:
        final_label = max(set(current_labels), key=current_labels.count) if current_labels else "O"
        avg_confidence = round((sum(current_confidences) / len(current_confidences)) * 100, 1) if current_confidences else 0.0
        clean_words.append({
            "word": current_word,
            "predicted_label": final_label,
            "confidence": f"{avg_confidence}%"
        })


    extracted_aspects = []
    for item in clean_words:

        if (item["predicted_label"] != "O" and 
            len(item["word"]) > 2):
            
            extracted_aspects.append({
                "aspect": item["word"],
                "sentiment_label": item["predicted_label"],
                "confidence": item["confidence"]
            })
            
    return {
        "text": request.text,
        "global_emotion": {
            "predicted_emotion": polish_emotion,
            "confidence": f"{emo_confidence}%"
        },
        "extracted_aspects": extracted_aspects,
        "full_sentence_annotations": clean_words
    }