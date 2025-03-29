from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

past_transactions = [
    {"description": "January electricity bill", "emissions_factor": "Electricity generation", "scope": 2},
    {"description": "Monthly internet subscription", "emissions_factor": "Data center services", "scope": 3},
    {"description": "Uber trip to client site", "emissions_factor": "Ridesharing services", "scope": 3},
    {"description": "Catering from Mario's Deli", "emissions_factor": "Food services", "scope": 1},
    {"description": "Printing business cards", "emissions_factor": "Paper manufacturing", "scope": 3},
    {"description": "Office cleaning service", "emissions_factor": "Facilities support services", "scope": 2},
    {"description": "March air travel to NYC", "emissions_factor": "Scheduled air transportation", "scope": 3},
    {"description": "Staff coffee beans", "emissions_factor": "Food and beverage retail", "scope": 1},
    {"description": "Office toilet paper", "emissions_factor": "Pulp and paper production", "scope": 3},
    {"description": "Software license renewal", "emissions_factor": "Cloud computing services", "scope": 3},
]

new_description = "All-hands lunch catering"

model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_transaction_label(new_description, past_transactions):
    if not past_transactions:
        return None

    df = pd.DataFrame(past_transactions)
    df["embedding"] = df["description"].apply(lambda desc: model.encode(desc))

    new_embedding = model.encode(new_description)
    df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity([emb], [new_embedding])[0][0])

    best_match = df.sort_values("similarity", ascending=False).iloc[0]

    return {
        "recommended_scope": best_match["scope"],
        "recommended_emissions_factor": best_match["emissions_factor"],
        "matched_description": best_match["description"],
        "similarity": best_match["similarity"]
    }

print(recommend_transaction_label(new_description, past_transactions))