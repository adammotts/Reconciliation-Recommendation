from fastapi import APIRouter, Body
from app.schemas.line_item import Recommendation, RecommendationQuery
from app.models.sentence_transformer import sentence_transformer_model

router = APIRouter()

@router.get("/get-recommendation", response_model=Recommendation)
async def get_recommendation(recommendation_query: RecommendationQuery = Body(...)):
    return await sentence_transformer_model.get_recommendation(recommendation_query)
