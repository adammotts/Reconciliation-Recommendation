from app.schemas.line_item import Recommendation, RecommendationQuery
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SentenceTransformerModel:
    def __init__(self, model):
        self.model = model

    async def get_recommendation(
        self,
        recommendation_query: RecommendationQuery
    ):
        
        if not recommendation_query:
            return None
        
        past_transactions = recommendation_query.past_transactions
        new_description = recommendation_query.new_description

        df = pd.DataFrame([transaction.model_dump() for transaction in past_transactions])
        df["embedding"] = df["description"].apply(lambda desc: self.model.encode(desc))
        new_embedding = self.model.encode(new_description)

        df["similarity"] = df["embedding"].apply(
            lambda emb: cosine_similarity([emb], [new_embedding])[0][0]
        )

        best_match = df.sort_values("similarity", ascending=False).iloc[0]

        return Recommendation(
            recommended_scope=best_match["scope"],
            recommended_emissions_factor=best_match["emissions_factor"],
            matched_description=best_match["description"],
            similarity=best_match["similarity"]
        )

model = SentenceTransformer("all-MiniLM-L6-v2")
_ = model.encode("warmup")

sentence_transformer_model = SentenceTransformerModel(model)