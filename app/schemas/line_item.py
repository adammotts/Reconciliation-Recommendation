# app/schemas/bathroom.py
from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    recommended_scope: int
    recommended_emissions_factor: str
    matched_description: str
    similarity: float

    class Config:
        from_attributes = True

class LineItem(BaseModel):
    description: str
    emissions_factor: str
    scope: int

class RecommendationQuery(BaseModel):
    past_line_items: List[LineItem]
    new_description: str
