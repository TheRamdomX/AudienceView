from typing import List, Optional
import json
import contextlib
import io
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.data_loader import load_all_content
from src.recommender import Recommender
from src.user_porfile import (
    create_multi_domain_user_profile,
    get_user_liked_summary_multi,
)
from src.llm_justifier import LLMJustifier


DEFAULT_LIKED = ["Inception", "Echoes of Time", "Aurora Skies - Celestial Nights Tour"]
DATA_DIR = "Data"
IMAGE_URL = "https://audienceview.com/wp-content/uploads/sites/2/2023/07/82409324_10156870761928715_3719706415825158144_n.webp"


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

        liked_indices = catalog_df[catalog_df['title'].isin(liked_titles)].index.tolist()
        user_profile = create_multi_domain_user_profile(liked_indices, catalog_df)
        user_summary = get_user_liked_summary_multi(liked_indices, catalog_df)

        recommender = Recommender()
        recommender.load(catalog_df)
        candidates_df = recommender.recommend(liked_indices, user_profile, top_n=top_candidates)
    return catalog_df, candidates_df, user_summary, liked_indices


def _format_recommendations_from_df(df):
    out = []
    for _, row in df.iterrows():
        out.append({
            "name": row.get("title"),
            "description": row.get("description") or row.get("overview") or "",
            "image": IMAGE_URL,
        })
    return out


@app.get("/health")
async def health():
    return {"status": "ok"}



@app.post("/recommendations/json")
async def recommendations_json(req: RecommendRequest):
    liked = req.liked_titles or DEFAULT_LIKED
    catalog_df, candidates_df, user_summary, _ = _build_candidates(liked, req.top_candidates)
    if catalog_df is None or candidates_df is None or candidates_df.empty:
        return {"recommendations": []}

    justifier = LLMJustifier()
    llm_json = justifier.recommend_json(candidates_df, user_summary, top_n=req.top_n)
    parsed = json.loads(llm_json)

    # parsed is expected to be {"recommendations": [{"id":.., "title":.., "content_type":..}, ...]}
    recs = parsed.get("recommendations") if isinstance(parsed, dict) else parsed
    if not recs:
        # fallback to top rows
        return {"recommendations": _format_recommendations_from_df(candidates_df.head(req.top_n))}

    # map ids/titles from LLM output to full rows when possible
    mapped = []
    for r in recs:
        # try by id first
        if isinstance(r, dict) and r.get("id") is not None:
            row = candidates_df[candidates_df['id'] == r.get('id')]
        else:
            row = candidates_df[candidates_df['title'] == r.get('title')]
        if not row.empty:
            mapped.extend(_format_recommendations_from_df(row.head(1)))
        else:
            # fallback to minimal
            mapped.append({
                "name": r.get('title') or r.get('name'),
                "description": "",
                "image": IMAGE_URL,
            })

    return {"recommendations": mapped}


@app.post("/recommendations/text")
async def recommendations_text(req: RecommendRequest):
    liked = req.liked_titles or DEFAULT_LIKED
    catalog_df, candidates_df, user_summary, _ = _build_candidates(liked, req.top_candidates)
    if catalog_df is None or candidates_df is None or candidates_df.empty:
        return {"paragraph": "No se encontraron recomendaciones."}

    justifier = LLMJustifier()
    paragraph = justifier.recommend_paragraph(candidates_df, user_summary, top_n=req.top_n)
    return {"paragraph": paragraph}



@app.get("/recommendations/json")
async def recommendations_json_get(liked_titles: Optional[str] = None, top_n: int = 3, top_candidates: int = 10):
    """GET endpoint que acepta liked_titles coma-separadas"""
    liked = DEFAULT_LIKED
    if liked_titles:
        liked = [t.strip() for t in liked_titles.split(",") if t.strip()]
    req = RecommendRequest(liked_titles=liked, top_n=top_n, top_candidates=top_candidates)
    return await recommendations_json(req)


@app.get("/recommendations/text")
async def recommendations_text_get(liked_titles: Optional[str] = None, top_n: int = 3, top_candidates: int = 10):
    """GET endpoint que acepta liked_titles coma-separadas"""
    liked = DEFAULT_LIKED
    if liked_titles:
        liked = [t.strip() for t in liked_titles.split(",") if t.strip()]
    req = RecommendRequest(liked_titles=liked, top_n=top_n, top_candidates=top_candidates)
    return await recommendations_text(req)


@app.post("/recommendations/movies")
async def recommendations_movies(req: RecommendRequest):
    liked = req.liked_titles or DEFAULT_LIKED
    catalog_df, candidates_df, user_summary, _ = _build_candidates(liked, req.top_candidates)
    # Filtrar solo movies
    movie_candidates = candidates_df[candidates_df['content_type'] == 'movie']

    if movie_candidates.empty:
        return {"recommendations": []}

    justifier = LLMJustifier()
    llm_json = justifier.recommend_json(movie_candidates, user_summary, top_n=req.top_n)
    parsed = json.loads(llm_json)
    recs = parsed.get("recommendations") if isinstance(parsed, dict) else parsed
    if not recs:
        return {"recommendations": _format_recommendations_from_df(movie_candidates.head(req.top_n))}

    mapped = []
    for r in recs:
        if isinstance(r, dict) and r.get("id") is not None:
            row = movie_candidates[movie_candidates['id'] == r.get('id')]
        else:
            row = movie_candidates[movie_candidates['title'] == r.get('title')]
        if not row.empty:
            mapped.extend(_format_recommendations_from_df(row.head(1)))
        else:
            mapped.append({
                "name": r.get('title') or r.get('name'),
                "description": "",
                "image": IMAGE_URL,
            })

    return {"recommendations": mapped}


@app.get("/recommendations/movies")
async def recommendations_movies_get(liked_titles: Optional[str] = None, top_n: int = 3, top_candidates: int = 15):
    liked = DEFAULT_LIKED
    if liked_titles:
        liked = [t.strip() for t in liked_titles.split(",") if t.strip()]
    req = RecommendRequest(liked_titles=liked, top_n=top_n, top_candidates=top_candidates)
    return await recommendations_movies(req)

#    uvicorn main:app --reload --host 0.0.0.0 --port 8000


if __name__ == "__main__":
        """Run the API with HTTPS using uvicorn and SSL certs from environment variables.

        Required env vars:
            - SSL_CERTFILE: path to the certificate file (e.g., cert.pem)
            - SSL_KEYFILE: path to the private key file (e.g., key.pem)

        Optional env vars:
            - HOST (default 0.0.0.0)
            - PORT (default 8443)
            - RELOAD (default true)
        """
        load_dotenv()
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8443"))
        reload = os.getenv("RELOAD", "true").lower() == "true"
        certfile = os.getenv("SSL_CERTFILE")
        keyfile = os.getenv("SSL_KEYFILE")

        if not certfile or not keyfile:
                raise RuntimeError(
                        "Faltan SSL_CERTFILE y/o SSL_KEYFILE para iniciar en HTTPS. "
                        "Consulta README.md para generar un certificado auto-firmado."
                )

        print(f"Iniciando API en https://{host}:{port} …")
        uvicorn.run(
                "api:app",
                host=host,
                port=port,
                reload=reload,
                ssl_certfile=certfile,
                ssl_keyfile=keyfile,
        )
