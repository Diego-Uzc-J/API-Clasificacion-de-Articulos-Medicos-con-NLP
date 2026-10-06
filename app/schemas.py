from typing import List, Optional
from pydantic import BaseModel, Field

class ArticleInput(BaseModel):
    titulo: str = Field(..., description="Título del artículo médico")
    resumen: str = Field(..., description="Resumen o abstract del artículo médico")

class PredictionOutput(BaseModel):
    categorias: List[str] = Field(..., description="Categorías predichas (cardiovascular, hepatorenal, oncológico, neurológico)")
    probabilidades: dict = Field(..., description="Puntuación de probabilidad para cada categoría")
