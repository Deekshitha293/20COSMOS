from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz  # Import fuzzywuzzy
import torch

app = Flask(__name__)

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample fund database
fund_data = {
    "Axis Long Term Equity Fund": {
        "type": "ELSS",
        "category": "Tax Saving",
        "sector": "Multi-cap",
        "description": "Offers tax benefits under section 80C"
    },
    "ICICI Tax Saving Plan": {
        "type": "ELSS",
        "category": "Tax Saving",
        "sector": "Large-cap",
        "description": "Save taxes under 80C with ELSS"
    },
    "HDFC Balanced Advantage Fund": {
        "type": "Balanced",
        "category": "Hybrid",
        "sector": "Balanced",
        "description": "Auto-adjusts equity-debt mix"
    }
}

# Fund names and embeddings
fund_names = list(fund_data.keys())
fund_texts = [
    f"{name} {meta['type']} {meta['category']} {meta['sector']} {meta['description']}"
    for name, meta in fund_data.items()
]
fund_embeddings = model.encode(fund_texts, convert_to_tensor=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    explanation = ""

    if request.method == 'POST':
        query = request.form['query']
        enriched_query = query.lower()
        query_embedding = model.encode(enriched_query, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, fund_embeddings)
        base_scores = similarities[0].tolist()

        # Get top 3 candidates for reranking
        top_3_indices = sorted(range(len(base_scores)), key=lambda i: base_scores[i], reverse=True)[:3]
        reranked_scores = []

        for i in top_3_indices:
            name = fund_names[i]
            meta = fund_data[name]
            score = base_scores[i]
            explanation += f"<b>🧐 Checking: {name}</b><br>Initial Score: {score:.2f}<br>"

            # Add bonus based on metadata matching (fuzzy matching)
            match_count = 0
            for key in ['type', 'category', 'sector']:
                # Apply fuzzy matching on metadata
                metadata_value = meta[key].lower()
                fuzzy_score = fuzz.partial_ratio(enriched_query, metadata_value)
                if fuzzy_score > 70:  # You can adjust the threshold as needed
                    score += 0.05
                    match_count += 1
                    explanation += f"✔ Matched <b>{key}</b>: {meta[key]} with fuzzy score {fuzzy_score}<br>"

            explanation += f"🎯 Final Adjusted Score: {score:.2f}<br><hr>"
            reranked_scores.append((i, score))

        # Select best reranked result
        best_index, best_score = max(reranked_scores, key=lambda x: x[1])

        if best_score < 0.3:
            result = "⚠ No strong match found. Try refining your query."
        else:
            result = {
                "name": fund_names[best_index],
                "metadata": fund_data[fund_names[best_index]],
                "score": round(best_score, 2)
            }

        explanation += f"<b>✅ Best Match:</b> {fund_names[best_index]}<br><b>Final Score:</b> {best_score:.2f}"

    return render_template("index.html", result=result, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
