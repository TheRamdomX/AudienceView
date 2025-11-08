import pandas as pd

def create_feature_soup(df: pd.DataFrame) -> pd.Series:
    """
    Combina características textuales de las películas en una sola cadena de texto ("soup").
    Esta cadena se usará para generar los embeddings.

    Args:
        df (pd.DataFrame): DataFrame con la información de las películas.
                           Debe contener 'title', 'overview', 'genres', y 'keywords'.

    Returns:
        pd.Series: Una serie de Pandas donde cada elemento es la cadena de texto combinada
                   para cada película.
    """
    # Asegurarse de que las columnas de listas se traten como tales y se unan en strings
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df['keywords'] = df['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

    # Combinar todas las características en una sola cadena (soup)
    soup = df['title'] + ' ' + df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']
    
    print("Creada la 'feature soup' para cada película.")
    return soup
