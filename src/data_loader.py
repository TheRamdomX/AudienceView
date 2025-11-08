import os
import json
import pandas as pd
from typing import List, Dict

def load_movies_from_json(filepath: str) -> pd.DataFrame:
    """
    Carga los datos de las películas desde un archivo JSON a un DataFrame de Pandas.

    Args:
        filepath (str): La ruta al archivo JSON.

    Returns:
        pd.DataFrame: Un DataFrame que contiene los datos de las películas.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Cargados {len(df)} películas desde {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocurrió un error al cargar o procesar el archivo JSON: {e}")
        return pd.DataFrame()


def _ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x is None:
        return []
    return [x]


def load_all_content(data_dir: str = 'Data') -> pd.DataFrame:
    """
    Carga y normaliza películas, canciones, merch, eventos teatrales y conciertos
    en un único DataFrame con esquema estándar:
    [id, title, content_type, genres(list), keywords(list), description(str)]
    """
    items = []

    # Movies
    movies_path = os.path.join(data_dir, 'movies.json')
    if os.path.exists(movies_path):
        try:
            df = pd.read_json(movies_path)
            for _, r in df.iterrows():
                items.append({
                    'id': r.get('id'),
                    'title': r.get('title'),
                    'content_type': 'movie',
                    'genres': r.get('genres', []) or [],
                    'keywords': r.get('keywords', []) or [],
                    'description': r.get('overview', '')
                })
        except Exception as e:
            print(f"No se pudo cargar movies: {e}")

    # Songs
    songs_path = os.path.join(data_dir, 'songs.json')
    if os.path.exists(songs_path):
        try:
            df = pd.read_json(songs_path)
            for _, r in df.iterrows():
                desc = f"{r.get('artist', '')} - {r.get('album', '')} ({r.get('year', '')}). {r.get('title', '')}"
                items.append({
                    'id': r.get('id'),
                    'title': r.get('title'),
                    'content_type': 'song',
                    'genres': r.get('genres', []) or [],
                    'keywords': r.get('keywords', []) or [],
                    'description': desc.strip()
                })
        except Exception as e:
            print(f"No se pudo cargar songs: {e}")

    # Merch
    merch_path = os.path.join(data_dir, 'merch.json')
    if os.path.exists(merch_path):
        try:
            df = pd.read_json(merch_path)
            for _, r in df.iterrows():
                category = r.get('category')
                items.append({
                    'id': r.get('id'),
                    'title': r.get('name'),
                    'content_type': 'merch',
                    'genres': [category] if isinstance(category, str) and category else [],
                    'keywords': r.get('keywords', []) or [],
                    'description': r.get('description', '')
                })
        except Exception as e:
            print(f"No se pudo cargar merch: {e}")

    # Theater events
    theater_path = os.path.join(data_dir, 'theater_events.json')
    if os.path.exists(theater_path):
        try:
            df = pd.read_json(theater_path)
            for _, r in df.iterrows():
                genre = r.get('genre')
                items.append({
                    'id': r.get('id'),
                    'title': r.get('title'),
                    'content_type': 'theater_event',
                    'genres': [genre] if isinstance(genre, str) and genre else [],
                    'keywords': r.get('keywords', []) or [],
                    'description': r.get('description', '')
                })
        except Exception as e:
            print(f"No se pudo cargar theater_events: {e}")

    # Concerts
    concerts_path = os.path.join(data_dir, 'concerts.json')
    if os.path.exists(concerts_path):
        try:
            df = pd.read_json(concerts_path)
            for _, r in df.iterrows():
                title = f"{r.get('artist', '')} - {r.get('tour_name', '')}".strip(' -')
                desc = f"{r.get('artist', '')} en {r.get('venue', '')}, {r.get('city', '')} ({r.get('date', '')})."
                items.append({
                    'id': r.get('id'),
                    'title': title,
                    'content_type': 'concert',
                    'genres': r.get('genres', []) or [],
                    'keywords': r.get('keywords', []) or [],
                    'description': desc.strip()
                })
        except Exception as e:
            print(f"No se pudo cargar concerts: {e}")

    catalog = pd.DataFrame(items)
    if not catalog.empty:
        # Normalizar columnas faltantes a listas/cadenas
        catalog['genres'] = catalog['genres'].apply(_ensure_list)
        catalog['keywords'] = catalog['keywords'].apply(_ensure_list)
        catalog['description'] = catalog['description'].fillna('')
        print(f"Catálogo unificado cargado: {len(catalog)} items de múltiples dominios.")
    else:
        print("No se encontraron items en el catálogo unificado.")

    return catalog
