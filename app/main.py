from fastapi import FastAPI, HTTPException
from app.schemas import ArticleInput, PredictionOutput
from app.pipeline import MedicalClassifierPipeline

app = FastAPI(
    title="API para Clasificación de Artículos Médicos con NLP",
    description="Clasificador multietiqueta de artículos médicos usando ALBERT (ONNX) optimizado para Render Free (512MB RAM)",
    version="1.0.0"
)

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = MedicalClassifierPipeline(model_dir="model")
    except Exception as e:
        print(f"Warning: Could not load ONNX model on startup: {e}")

@app.get("/")
def read_root():
    return {"message": "API de Clasificación de Artículos Médicos funcionando correctamente", "status": "active"}

@app.post("/predict", response_model=PredictionOutput)
def predict_article(article: ArticleInput):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado o no disponible en entorno de pruebas.")
    try:
        result = pipeline.predict(article.titulo, article.resumen)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
