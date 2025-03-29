from app.schemas.line_item import Recommendation, RecommendationQuery
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pandas as pd

class SentenceTransformerModel:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    async def get_recommendation(
        self,
        recommendation_query: RecommendationQuery
    ) -> List[Recommendation]:
        
        if not recommendation_query.past_transactions or not recommendation_query.unreconciled_transactions:
            return []

        past_transactions = recommendation_query.past_transactions
        unreconciled_transactions = recommendation_query.unreconciled_transactions

        past_descriptions = [t.description for t in past_transactions]
        past_embeddings = self.model.encode(past_descriptions, convert_to_numpy=True)

        past_df = pd.DataFrame([
            {
                "description": t.description,
                "scope": t.scope,
                "emissions_factor": t.emissions_factor,
                "emission_factor_id": t.emissions_factor_id,
                "embedding": emb
            }
            for t, emb in zip(past_transactions, past_embeddings)
        ])

        new_descriptions = [u.description for u in unreconciled_transactions]
        new_ids = [u.id for u in unreconciled_transactions]
        new_embeddings = self.model.encode(new_descriptions, convert_to_numpy=True)

        recommendations = []
        for i, new_embedding in enumerate(new_embeddings):
            similarities = cosine_similarity(past_embeddings, [new_embedding]).flatten()
            past_df["similarity"] = similarities

            best_match = past_df.loc[past_df["similarity"].idxmax()]

            recommendations.append(Recommendation(
                id=new_ids[i],
                recommended_scope=best_match["scope"],
                recommended_emissions_factor=best_match["emissions_factor"],
                recommended_emissions_factor_id=best_match["emission_factor_id"],
                matched_description=best_match["description"],
                similarity=best_match["similarity"]
            ))

        return recommendations


model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
_ = model.encode("warmup")

sentence_transformer_model = SentenceTransformerModel(model)