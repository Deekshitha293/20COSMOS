from sentence_transformers import SentenceTransformer, util
import pandas as pd

class FundMatcher:
    def __init__(self, fund_data_path):
        self.df = pd.read_csv(fund_data_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.fund_names = self.df['scheme_name'].astype(str).tolist()
        self.embeddings = self.model.encode(self.fund_names, convert_to_tensor=True)

    def match(self, query, top_k=3):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        top_results = scores.topk(k=top_k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            fund_info = self.df.iloc[int(idx)]
            results.append({
                'name': fund_info['scheme_name'],
                'score': float(score),
                'category': fund_info.get('category', ''),
                'sub_category': fund_info.get('sub_category', ''),
                'amc': fund_info.get('amc_name', '')
            })
        return results
