from fastapi import APIRouter, Body
from app.schemas.line_item import Recommendation, RecommendationQuery
from app.models.sentence_transformer import sentence_transformer_model
from typing import List

router = APIRouter()

@router.get("/get-recommendation", response_model=List[Recommendation])
async def get_recommendation(recommendation_query: RecommendationQuery = Body(...)):
    return await sentence_transformer_model.get_recommendation(recommendation_query)
