import argparse
import json
import contextlib
import io
from src.data_loader import load_all_content
from src.recommender import Recommender
from src.llm_justifier import LLMJustifier
from src.user_porfile import (
    create_multi_domain_user_profile,
    get_user_liked_summary_multi,
)


def main():
    """Flujo multi-dominio con opciones de salida --json o --text (LLM genera la recomendación)."""
    parser = argparse.ArgumentParser(description="Recomendador multi-dominio")
    parser.add_argument("--json", dest="as_json", action="store_true", help="LLM devuelve JSON con 3 recomendaciones")
    parser.add_argument("--text", dest="as_text", action="store_true", help="LLM devuelve un párrafo con 3 recomendaciones")
    args = parser.parse_args()

    liked_titles = ["Inception", "Echoes of Time", "Aurora Skies - Celestial Nights Tour"]

    # MODO JSON
    if args.as_json:
        silent_buffer = io.StringIO()
        with contextlib.redirect_stdout(silent_buffer):
            catalog_df = load_all_content('Data')

            liked_indices = catalog_df[catalog_df['title'].isin(liked_titles)].index.tolist()
            user_profile = create_multi_domain_user_profile(liked_indices, catalog_df)
            user_summary = get_user_liked_summary_multi(liked_indices, catalog_df)
            recommender = Recommender()
            recommender.load(catalog_df)
            candidates_df = recommender.recommend(liked_indices, user_profile, top_n=10)
            
            justifier = LLMJustifier()
            llm_json = justifier.recommend_json(candidates_df, user_summary, top_n=3)

        print(llm_json)
        return



    # MODO TEXTO
    
    silent_buffer = io.StringIO()
    with contextlib.redirect_stdout(silent_buffer):
        catalog_df = load_all_content('Data')

        liked_indices = catalog_df[catalog_df['title'].isin(liked_titles)].index.tolist()
        user_profile = create_multi_domain_user_profile(liked_indices, catalog_df)
        user_summary = get_user_liked_summary_multi(liked_indices, catalog_df)
        recommender = Recommender()
        recommender.load(catalog_df)
        candidates_df = recommender.recommend(liked_indices, user_profile, top_n=10)

        justifier = LLMJustifier()
        paragraph = justifier.recommend_paragraph(candidates_df, user_summary, top_n=3)
    print(paragraph)


if __name__ == "__main__":
    main()
