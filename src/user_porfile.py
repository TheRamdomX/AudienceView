import numpy as np
import pandas as pd
from typing import List, Dict

def create_user_profile(liked_movies_indices: List[int], movie_embeddings: np.ndarray, movies_df: pd.DataFrame) -> np.ndarray:
    """
    Crea un perfil de usuario promediando los embeddings de las películas que le han gustado.

    Args:
        liked_movies_indices (List[int]): Lista de índices de las películas que le han gustado al usuario.
        movie_embeddings (np.ndarray): Array de NumPy con todos los embeddings de películas.
        movies_df (pd.DataFrame): DataFrame con los datos de las películas.

    Returns:
        np.ndarray: El vector de perfil de usuario.
    """
    if not liked_movies_indices:
        print("Advertencia: No se proporcionaron películas gustadas para crear el perfil.")
        return np.zeros(movie_embeddings.shape[1])

    liked_embeddings = movie_embeddings[liked_movies_indices]
    user_profile_vector = np.mean(liked_embeddings, axis=0)

    liked_titles = movies_df.loc[liked_movies_indices, 'title'].tolist()
    print(f"Perfil de usuario creado a partir de las películas: {', '.join(liked_titles)}")
    
    return user_profile_vector

def get_user_liked_movies_summary(liked_movies_indices: List[int], movies_df: pd.DataFrame) -> str:
    """
    Genera un resumen de los gustos del usuario basado en las películas que le gustaron.
    
    Args:
        liked_movies_indices (List[int]): Índices de las películas que gustaron.
        movies_df (pd.DataFrame): DataFrame de películas.

    Returns:
        str: Un resumen textual de los gustos del usuario.
    """
    liked_movies_df = movies_df.loc[liked_movies_indices]
    
    genres = liked_movies_df['genres'].explode().unique()
    keywords = liked_movies_df['keywords'].explode().unique()

    summary = (
        f"Al usuario le gustan películas con los siguientes géneros: {', '.join(genres)}. "
        f"También muestra interés en temas como: {', '.join(keywords)}."
    )
    return summary

def create_fake_user_profile(liked_movies_indices: List[int], movies_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Crea un perfil de usuario falso basado en la agregación (unión) de géneros y keywords
    de las películas que le han gustado. Útil para un MVP sin embeddings.

    Returns:
        Dict[str, List[str]]: Diccionario con listas de 'genres' y 'keywords'.
    """
    if not liked_movies_indices:
        return {"genres": [], "keywords": []}

    liked_movies_df = movies_df.loc[liked_movies_indices]
    genres = sorted(set(g for lst in liked_movies_df['genres'] for g in lst))
    keywords = sorted(set(k for lst in liked_movies_df['keywords'] for k in lst))
    return {"genres": genres, "keywords": keywords}

def create_multi_domain_user_profile(liked_indices: List[int], catalog_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Agrega géneros y keywords de cualquier tipo de contenido gustado."""
    if not liked_indices:
        return {"genres": [], "keywords": []}
    liked_df = catalog_df.loc[liked_indices]
    genres = sorted(set(g for lst in liked_df['genres'] for g in lst))
    keywords = sorted(set(k for lst in liked_df['keywords'] for k in lst))
    return {"genres": genres, "keywords": keywords}


def get_user_liked_summary_multi(liked_indices: List[int], catalog_df: pd.DataFrame) -> str:
    """Genera resumen incluyendo tipos de contenido."""
    if not liked_indices:
        return "El usuario aún no ha marcado contenidos favoritos."
    liked_df = catalog_df.loc[liked_indices]
    types = ', '.join(sorted(liked_df['content_type'].unique()))
    genres = ', '.join(sorted(set(g for lst in liked_df['genres'] for g in lst)))
    keywords = ', '.join(sorted(set(k for lst in liked_df['keywords'] for k in lst)))
    return (
        f"El usuario ha mostrado interés en tipos: {types}. Géneros frecuentes: {genres}. "
        f"Temas/keywords: {keywords}."
    )