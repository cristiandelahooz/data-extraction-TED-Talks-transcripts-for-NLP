"""
Módulo para procesamientO
"""

import pandas as pd
import numpy as np
from collections import Counter
from textblob import TextBlob
from tqdm.auto import tqdm
from .progress_tracker import ProgressTracker, real_time_feedback

try:
    import spacy
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    print(f"Warning: Some NLP libraries not available: {e}")


def process_text_features(df, text_column='transcript_clean', nlp_models=None):
    """
    Procesa todas las características de texto para un DataFrame
    """
    # Inicializar tracker de progreso
    tracker = ProgressTracker(total_steps=5, description="Procesamiento NLP")
    tracker.start("Iniciando extracción de características NLP")
    
    real_time_feedback(f"Procesando columna: {text_column}")
    
    if text_column not in df.columns:
        print(f"Columna {text_column} no encontrada")
        return df
    
    # 1. CARGAR MODELOS
    tracker.step("Cargando modelos de NLP")
    if nlp_models is None:
        nlp_models = load_nlp_models()
        real_time_feedback("Modelos de NLP cargados")
    
    # 2. PREPARAR MUESTRA
    tracker.step("Preparando muestra de datos")
    sample_size = min(100, len(df))
    real_time_feedback(f"Procesando muestra de {sample_size} textos para velocidad...")
    sample_df = df.head(sample_size).copy()
    
    # 3. PROCESAR SENTIMIENTOS
    tracker.step("Analizando sentimientos con TextBlob")
    sample_df = _process_sentiments(sample_df, text_column)
    real_time_feedback("Análisis de sentimientos completado")
    
    # 4. PROCESAR CARACTERÍSTICAS TEXTUALES
    tracker.step("Extrayendo características textuales")
    sample_df = _process_text_features(sample_df, text_column, nlp_models)
    real_time_feedback("Características textuales extraídas")
    
    # 5. PROCESAR ENTIDADES NOMBRADAS
    tracker.step("Identificando entidades nombradas")
    if nlp_models['spacy'] is not None:
        sample_df = process_named_entities(sample_df, text_column, nlp_models['spacy'], sample_size=sample_size)
        real_time_feedback("Entidades nombradas identificadas")
    else:
        real_time_feedback("spaCy no disponible - omitiendo entidades nombradas")
    
    # Mostrar estadísticas finales
    _show_feature_statistics(sample_df)
    
    tracker.finish("Procesamiento NLP completado")
    
    return sample_df

def load_nlp_models():
    """Carga modelos de NLP necesarios"""
    models = {}
    
    # Cargar spaCy
    try:
        models['spacy'] = spacy.load("en_core_web_sm")
        print("✓ spaCy cargado correctamente")
    except Exception as e:
        print(f"⚠ Error cargando spaCy: {e}")
        models['spacy'] = None
    
    # Configurar NLTK
    try:
        models['stop_words'] = set(stopwords.words('english'))
        models['lemmatizer'] = WordNetLemmatizer()
        print("✓ NLTK configurado correctamente")
    except Exception as e:
        print(f"⚠ Error configurando NLTK: {e}")
        models['stop_words'] = set()
        models['lemmatizer'] = None
    
    return models

def _process_sentiments(df, text_column):
    """Procesa análisis de sentimientos"""
    real_time_feedback("Analizando polaridad y subjetividad...")
    tqdm.pandas(desc="Sentimientos", leave=False)
    sentiment_results = df[text_column].progress_apply(analyze_sentiment)
    
    # Convertir resultados a columnas
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    for col in sentiment_df.columns:
        df[f'sentiment_{col}'] = sentiment_df[col]

    return df

def analyze_sentiment(text):
    """
    Análisis híbrido TextBlob + VADER para máxima precisión
    Combina lo mejor de ambos para TED Talks
    """
    if pd.isna(text) or text == '':
        return _get_empty_sentiment()
    
    try:
        # TextBlob: Para análisis general
        blob = TextBlob(text)
        tb_sentiment = blob.sentiment
        
        # VADER: Para texto informal y énfasis
        from nltk.sentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        vader_scores = vader.polarity_scores(text)
        
        # Combinar ambos análisis
        combined_polarity = (tb_sentiment.polarity + vader_scores['compound']) / 2
        combined_subjectivity = tb_sentiment.subjectivity
        
        # Usar VADER para clasificación (mejor con texto hablado)
        if vader_scores['compound'] >= 0.5:
            sentiment_label = 'muy_positivo'
        elif vader_scores['compound'] >= 0.1:
            sentiment_label = 'positivo'
        elif vader_scores['compound'] <= -0.5:
            sentiment_label = 'muy_negativo'
        elif vader_scores['compound'] <= -0.1:
            sentiment_label = 'negativo'
        else:
            sentiment_label = 'neutral'
        
        # RETORNAR CON CLAVES LIMPIAS (sin prefijo sentiment_)
        return {
            'polarity': combined_polarity,                    # ✓
            'subjectivity': combined_subjectivity,            # ✓
            'label': sentiment_label,                         # ✓
            'vader_compound': vader_scores['compound'],       # ✓
            'vader_positive': vader_scores['pos'],            # ✓
            'vader_negative': vader_scores['neg'],            # ✓
            'vader_neutral': vader_scores['neu'],             # ✓
            'agreement': _calculate_agreement(tb_sentiment.polarity, vader_scores['compound'])  # ✓
        }
    
    except Exception as e:
        print(f"Error en análisis híbrido: {e}")
        return _get_empty_sentiment()

def _calculate_agreement(textblob_score, vader_score):
    """Calcula qué tan de acuerdo están ambos algoritmos"""
    if (textblob_score > 0 and vader_score > 0) or \
       (textblob_score < 0 and vader_score < 0) or \
       (abs(textblob_score) < 0.1 and abs(vader_score) < 0.1):
        return 1.0  # Están de acuerdo
    else:
        return 0.0  # Desacuerdo

def _get_empty_sentiment():
    """Valores por defecto para textos vacíos"""
    return {
        'polarity': 0.0,
        'subjectivity': 0.0,
        'label': 'neutral',
        'vader_compound': 0.0,
        'vader_positive': 0.0,
        'vader_negative': 0.0,
        'vader_neutral': 1.0,
        'agreement': 1.0
    }

def process_named_entities(df, text_column, spacy_model, sample_size=None):
    """
    Procesa entidades nombradas para una columna de texto en el DataFrame.
    
    Args:
        df: DataFrame de Pandas
        text_column: Nombre de la columna con texto a procesar
        spacy_model: Modelo de spaCy cargado
        sample_size: Número de filas a procesar (None para procesar todas)
    
    Returns:
        DataFrame con nuevas columnas para conteos de entidades
    """
    # Determinar tamaño de muestra
    entity_sample_size = len(df) if sample_size is None else min(sample_size, len(df))
    print(f"Procesando {entity_sample_size} textos para entidades nombradas...")
    
    # Habilitar progress_apply
    tqdm.pandas(desc="Entidades")
    
    # Procesar entidades nombradas
    entity_results = df[text_column].head(entity_sample_size).progress_apply(
        lambda x: extract_named_entities(x, spacy_model)
    )
    
    # Procesar conteos de entidades
    entity_counts_list = [count_entity_types(entities) for entities in entity_results]
    entity_df = pd.DataFrame(entity_counts_list, index=df.index[:entity_sample_size])
    
    # Inicializar columnas de conteo en el DataFrame original
    result_df = df.copy()  # Evitar modificar el DataFrame original
    for col in entity_df.columns:
        result_df[col] = 0  # Inicializar con ceros
    
    # Actualizar valores para las filas procesadas
    result_df.iloc[:entity_sample_size][entity_df.columns] = entity_df
    
    return result_df

def extract_named_entities(text, nlp_model):
    """
    Extrae entidades nombradas usando spaCy
    """
    if pd.isna(text) or text == '' or nlp_model is None:
        print("Texto vacío o modelo spaCy no disponible")
        return {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'MONEY': [],
            'DATE': [],
            'TIME': [],
            'PERCENT': [],
            'QUANTITY': []
        }
    
    # Limitar texto para eficiencia
    text_limited = text[:1_000_000] if len(text) > 1_000_000 else text
    try:
        doc = nlp_model(text_limited)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],      # Geopolitical entities
            'MONEY': [],
            'DATE': [],
            'TIME': [],
            'PERCENT': [],
            'QUANTITY': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.lower().strip())
        
        # Eliminar duplicados manteniendo orden
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    except Exception as e:
        print(f"Error procesando texto: {e}")
        return {
            'PERSON': [], 'ORG': [], 'GPE': [], 'MONEY': [],
            'DATE': [], 'TIME': [], 'PERCENT': [], 'QUANTITY': []
        }

def count_entity_types(entities_dict):
    """
    Cuenta el número de entidades por tipo
    """
    counts = {}
    for ent_type, ent_list in entities_dict.items():
        counts[f'{ent_type.lower()}_count'] = len(ent_list)
    return counts

def _process_text_features(df, text_column, nlp_models):
    """Procesa características textuales básicas"""
    real_time_feedback("Calculando longitud, palabras, oraciones...")
    tqdm.pandas(desc="Características", leave=False)
    text_features = df[text_column].progress_apply(
        lambda x: extract_text_features(x, nlp_models['stop_words'])
    )
    
    # Convertir a columnas
    features_df = pd.DataFrame(text_features.tolist())
    for col in features_df.columns:
        df[f'text_{col}'] = features_df[col]
    
    return df

def extract_text_features(text, stop_words=None):
    """
    Extrae características textuales básicas
    """
    if pd.isna(text) or text == '':
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'unique_words': 0,
            'lexical_diversity': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0
        }
    
    # Contar palabras y oraciones
    words = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.split()
    sentences = sent_tokenize(text) if 'sent_tokenize' in globals() else text.split('.')
    
    # Filtrar stop words si están disponibles
    if stop_words:
        words = [word for word in words if word not in stop_words and word.isalpha()]
    else:
        words = [word for word in words if word.isalpha()]
    
    # Calcular métricas
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Contar signos de puntuación
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Ratio de mayúsculas
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'unique_words': unique_words,
        'lexical_diversity': lexical_diversity,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': uppercase_ratio
    }

def _show_feature_statistics(df):
    """Muestra estadísticas de las características procesadas"""
    print("\n=== ESTADÍSTICAS DE CARACTERÍSTICAS ===")
    
    # Sentimientos
    if 'sentiment_polarity' in df.columns:
        polarity_stats = df['sentiment_polarity'].describe()
        print("Polaridad de sentimiento:")
        print(f"  Media: {polarity_stats['mean']:.3f}")
        print(f"  Rango: [{polarity_stats['min']:.3f}, {polarity_stats['max']:.3f}]")
        
        sentiment_dist = df['sentiment_label'].value_counts()
        print("Distribución de sentimientos:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    # Características textuales
    text_feature_columns = [col for col in df.columns if col.startswith('text_')]
    if text_feature_columns:
        print("\nCaracterísticas textuales promedio:")
        for col in text_feature_columns:
            mean_val = df[col].mean()
            print(f"  {col.replace('text_', '')}: {mean_val:.2f}")
    
    # Entidades
    entity_columns = [col for col in df.columns if col.endswith('_count')]
    if entity_columns:
        print("\nEntidades promedio por texto:")
        for col in entity_columns:
            mean_val = df[col].mean()
            entity_type = col.replace('_count', '').upper()
            print(f"  {entity_type}: {mean_val:.2f}")

def get_text_statistics(df, text_columns=None):
    """
    Obtiene estadísticas generales de las columnas de texto
    """
    if text_columns is None:
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    stats = {}
    
    for col in text_columns:
        if col in df.columns:
            col_stats = {
                'total_texts': len(df),
                'non_empty_texts': df[col].notna().sum(),
                'avg_length': df[col].str.len().mean(),
                'median_length': df[col].str.len().median(),
                'max_length': df[col].str.len().max(),
                'min_length': df[col].str.len().min()
            }
            stats[col] = col_stats
    
    return stats

def create_word_frequency_analysis(df, text_column, top_n=20, stop_words=None):
    """
    Crea análisis de frecuencia de palabras para todo el corpus
    """
    print("=== ANÁLISIS DE FRECUENCIA DE PALABRAS ===")
    
    if text_column not in df.columns:
        print(f"⚠ Columna {text_column} no encontrada")
        return {}
    
    # Combinar todos los textos
    all_text = ' '.join(df[text_column].fillna('').astype(str))
    
    # Extraer palabras clave del corpus completo
    keywords = extract_keywords(all_text, n_keywords=top_n, stop_words=stop_words)
    
    print(f"Top {top_n} palabras más frecuentes:")
    for word, freq in keywords:
        print(f"  {word}: {freq}")
    
    return dict(keywords)

def extract_keywords(text, n_keywords=10, stop_words=None):
    """
    Extrae palabras clave más frecuentes del texto
    """
    if pd.isna(text) or text == '':
        return []
    
    # Limpiar y tokenizar
    words = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.lower().split()
    
    # Filtrar palabras
    if stop_words:
        words = [word for word in words if word not in stop_words and word.isalpha() and len(word) > 3]
    else:
        words = [word for word in words if word.isalpha() and len(word) > 3]
    
    # Contar frecuencias
    word_freq = Counter(words)
    
    # Retornar top keywords
    return word_freq.most_common(n_keywords)