import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI

class LLMJustifier:
    """
    Genera justificaciones y recomendaciones usando la API de OpenAI.
    Variables de entorno:
      OPENAI_API_KEY (requerida)
      OPENAI_MODEL o GPT_MODEL (modelo, por defecto 'gpt-4o-mini')
    """

    def __init__(self, model_name: Optional[str] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en entorno / .env")
        env_model = os.getenv("OPENAI_MODEL") or os.getenv("GPT_MODEL")
        self.model = (model_name or env_model or "gpt-4o-mini").strip()
        self.client = OpenAI(api_key=api_key)
        print(f"Cliente OpenAI inicializado. Modelo: {self.model}")

    def _chat(self, system_msg: str, user_msg: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error API OpenAI: {e}")
            return ""

    def justify(self, recommendations_df: pd.DataFrame, user_summary: str) -> str:
        if recommendations_df.empty:
            return "No hay recomendaciones para justificar."
        prompt = self._build_prompt(recommendations_df, user_summary)
        system_msg = (
            "Eres un experto en contenido cultural (cine, música, eventos y productos). "
            "Explica de forma breve, amigable y convincente por qué este grupo de recomendaciones encaja "
            "con los gustos del usuario, referenciando géneros/temas relevantes."
        )
        out = self._chat(system_msg, prompt)
        return out or "No se pudo generar una justificación en este momento."

    def recommend_json(self, candidates_df: pd.DataFrame, user_summary: str, top_n: int = 3) -> str:
        if candidates_df.empty:
            return '{"recommendations": []}'
        prompt = self._build_prompt_for_json(candidates_df, user_summary, top_n)
        system_msg = (
            "Eres un sistema que devuelve estrictamente JSON válido. "
            f"Selecciona exactamente {top_n} recomendaciones del listado de candidatos. "
            "Usa el esquema: {\"recommendations\":[{\"id\":number,\"title\":string,\"content_type\":string}]}. "
            "Responde SOLO con JSON exacto, sin texto adicional."
        )
        out = self._chat(system_msg, prompt)
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict) and isinstance(parsed.get('recommendations'), list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return '{"recommendations": []}'

    def recommend_paragraph(self, candidates_df: pd.DataFrame, user_summary: str, top_n: int = 3) -> str:
        if candidates_df.empty:
            return "No hay recomendaciones disponibles."
        prompt = self._build_prompt_for_paragraph(candidates_df, user_summary, top_n)
        system_msg = (
            "Eres un experto en recomendaciones. "
            f"Escribe UN solo párrafo (4-6 líneas) en español, nombra explícitamente {top_n} títulos elegidos del listado y explica por qué encajan."
        )
        out = self._chat(system_msg, prompt)
        return out or "No se pudo generar la recomendación en este momento."

    def _build_prompt(self, recommendations_df: pd.DataFrame, user_summary: str) -> str:
        desc_col = 'overview' if 'overview' in recommendations_df.columns else 'description'
        ct_col = 'content_type' if 'content_type' in recommendations_df.columns else None
        lines = []
        for _, row in recommendations_df.iterrows():
            title = row.get('title', 'Item')
            desc = row.get(desc_col, '') if desc_col else ''
            if ct_col:
                ct = row.get(ct_col, '')
                lines.append(f"- [{ct}] {title}: {desc}")
            else:
                lines.append(f"- {title}: {desc}")
        recommended_str = "\n".join(lines)
        return (
            f"Basado en que {user_summary.lower()}, he seleccionado estas recomendaciones para ti:\n"
            f"{recommended_str}\n\n"
            "En 4-6 líneas, explica de forma concisa por qué este conjunto tiene sentido para el usuario."
        )

    def _build_prompt_for_json(self, candidates_df: pd.DataFrame, user_summary: str, top_n: int) -> str:
        desc_col = 'overview' if 'overview' in candidates_df.columns else 'description'
        rows = []
        for _, row in candidates_df.iterrows():
            desc = (row.get(desc_col, '') or '')
            if len(desc) > 220:
                desc = desc[:217] + '...'
            rows.append(
                f"- id: {row.get('id','')}; title: {row.get('title','')}; type: {row.get('content_type','')}; "
                f"genres: {', '.join(row.get('genres', []))}; keywords: {', '.join(row.get('keywords', []))}; desc: {desc}"
            )
        candidates_str = "\n".join(rows)
        return (
            "Usuario: " + user_summary + "\n" +
            f"Candidatos (elige exactamente {top_n} distintos y devuelve su id/title/content_type):\n" + candidates_str + "\n" +
            "Responde SOLO con JSON válido según el esquema indicado."
        )

    def _build_prompt_for_paragraph(self, candidates_df: pd.DataFrame, user_summary: str, top_n: int) -> str:
        desc_col = 'overview' if 'overview' in candidates_df.columns else 'description'
        rows = []
        for _, row in candidates_df.iterrows():
            desc = (row.get(desc_col, '') or '')
            if len(desc) > 160:
                desc = desc[:157] + '...'
            rows.append(
                f"- {row.get('title','')} [{row.get('content_type','')}]: {desc}"
            )
        candidates_str = "\n".join(rows)
        return (
            "Perfil del usuario: " + user_summary + "\n" +
            f"Del siguiente listado de candidatos, elige exactamente {top_n} y escribe UN párrafo en español mencionando los {top_n} títulos y por qué encajan.\n" +
            candidates_str
        )
