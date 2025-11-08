from typing import List, Optional
import json
import contextlib
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_all_content
from src.recommender import Recommender
from src.user_porfile import (
    create_multi_domain_user_profile,
    get_user_liked_summary_multi,
)
from src.llm_justifier import LLMJustifier


DEFAULT_LIKED = ["Inception", "Echoes of Time", "Aurora Skies - Celestial Nights Tour"]
DATA_DIR = "Data"

app = FastAPI(title="AudienceView Recommender API", version="0.1.0")


class RecommendRequest(BaseModel):
    liked_titles: Optional[List[str]] = None
    top_n: int = 3
    top_candidates: int = 10


def _build_candidates(liked_titles: List[str], top_candidates: int):
    # Silenciar prints internos del pipeline
    silent_buffer = io.StringIO()
    with contextlib.redirect_stdout(silent_buffer):
        catalog_df = load_all_content(DATA_DIR)
        if catalog_df.empty:
            return catalog_df, None, None, None

        liked_indices = catalog_df[catalog_df['title'].isin(liked_titles)].index.tolist()
        user_profile = create_multi_domain_user_profile(liked_indices, catalog_df)
        user_summary = get_user_liked_summary_multi(liked_indices, catalog_df)

        recommender = Recommender()
        recommender.load(catalog_df)
        candidates_df = recommender.recommend(liked_indices, user_profile, top_n=top_candidates)
    return catalog_df, candidates_df, user_summary, liked_indices


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recommendations/json")
async def recommendations_json(req: RecommendRequest):
    liked = req.liked_titles or DEFAULT_LIKED
    catalog_df, candidates_df, user_summary, _ = _build_candidates(liked, req.top_candidates)
    if catalog_df is None or candidates_df is None or candidates_df.empty:
        return {"recommendations": []}

    # LLM genera JSON (id, title, content_type)
    try:
        justifier = LLMJustifier()
        llm_json = justifier.recommend_json(candidates_df, user_summary, top_n=req.top_n)
        return json.loads(llm_json)
    except Exception as e:
        # Fallback simple
        fallback = []
        for _, row in candidates_df.head(req.top_n).iterrows():
            fallback.append({
                "id": int(row.get("id")) if row.get("id") is not None else None,
                "title": row.get("title"),
                "content_type": row.get("content_type"),
            })
        return {"recommendations": fallback}


@app.post("/recommendations/text")
async def recommendations_text(req: RecommendRequest):
    liked = req.liked_titles or DEFAULT_LIKED
    catalog_df, candidates_df, user_summary, _ = _build_candidates(liked, req.top_candidates)
    if catalog_df is None or candidates_df is None or candidates_df.empty:
        return {"paragraph": "No se encontraron recomendaciones."}

    try:
        justifier = LLMJustifier()
        paragraph = justifier.recommend_paragraph(candidates_df, user_summary, top_n=req.top_n)
        return {"paragraph": paragraph}
    except Exception as e:
        titles = ", ".join(candidates_df.head(req.top_n)['title'].tolist())
        return {
            "paragraph": f"Recomendamos {titles} porque comparten géneros y temas centrales con tus intereses previos.",
            "note": "fallback"
        }


@app.get("/recommendations/json")
async def recommendations_json_get(liked_titles: Optional[str] = None, top_n: int = 3, top_candidates: int = 10):
    """GET endpoint que acepta liked_titles coma-separadas"""
    liked = DEFAULT_LIKED
    if liked_titles:
        liked = [t.strip() for t in liked_titles.split(",") if t.strip()]
    req = RecommendRequest(liked_titles=liked, top_n=top_n, top_candidates=top_candidates)
    return await recommendations_json(req)


@app.get("/recommendations/text")
async def recommendations_text_get(liked_titles: Optional[str] = None, top_n: int = 1, top_candidates: int = 10):
    """GET endpoint que acepta liked_titles coma-separadas"""
    liked = DEFAULT_LIKED
    if liked_titles:
        liked = [t.strip() for t in liked_titles.split(",") if t.strip()]
    req = RecommendRequest(liked_titles=liked, top_n=top_n, top_candidates=top_candidates)
    return await recommendations_text(req)


#    uvicorn api:app --reload --host 0.0.0.0 --port 8000
