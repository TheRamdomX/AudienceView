import pandas as pd
from typing import List, Dict

class Recommender:
    """
    Recomendador simple basado en solapamiento de géneros y keywords.
    Funciona con catálogos multi-dominio (movies, songs, merch, theater, concerts).
    """

    def __init__(self):
        self.catalog_df = None
        print("Recomendador inicializado")

    def load(self, catalog_df: pd.DataFrame):
        """Carga el DataFrame unificado de contenido."""
        self.catalog_df = catalog_df.copy()
        print(f"Catálogo cargado: {len(self.catalog_df)} items.")

    def recommend(self, liked_indices: List[int], user_profile: Dict[str, List[str]], top_n: int = 5) -> pd.DataFrame:
        """Recomienda por coincidencia de géneros/keywords (géneros pesan 2x)."""
        if self.catalog_df is None or self.catalog_df.empty:
            return pd.DataFrame()

        df = self.catalog_df.copy()
        df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
        df['keywords'] = df['keywords'].apply(lambda x: x if isinstance(x, list) else [])

        liked_set = set(liked_indices)
        user_genres = set(user_profile.get('genres', []))
        user_keywords = set(user_profile.get('keywords', []))

        scores = []
        for idx, row in df.iterrows():
            if idx in liked_set:
                continue
            g_overlap = len(user_genres.intersection(row['genres']))
            k_overlap = len(user_keywords.intersection(row['keywords']))
            score = g_overlap * 2 + k_overlap
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [i for i, s in scores if s > 0][:top_n]
        recs = df.loc[top].copy()
        recs['score'] = [dict(scores)[i] for i in top]
        return recs[['id', 'title', 'content_type', 'score', 'genres', 'keywords', 'description']]
