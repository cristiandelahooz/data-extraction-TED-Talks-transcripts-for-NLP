#!/usr/bin/env python3
"""
Ejemplo de uso para crear gráficos de radar de sentimientos
"""

import pandas as pd
import numpy as np
from modules.visualizer import create_sentiment_radar_chart
from modules.nlp_processor import process_text_features

def create_sample_data():
    """
    Crea datos de ejemplo para demostrar el gráfico de radar
    """
    np.random.seed(42)
    
    # Crear datos simulados con diferentes años
    years = list(range(2010, 2021))
    data = []
    
    for year in years:
        for i in range(10):  # 10 muestras por año
            # Simular texto de ejemplo
            sample_text = f"This is a sample text from year {year}. It contains various emotions and sentiments."
            
            data.append({
                'year': year,
                'transcript_clean': sample_text,
                'views': np.random.randint(10000, 1000000)
            })
    
    return pd.DataFrame(data)

def run_sentiment_analysis_example():
    """
    Ejecuta un ejemplo completo de análisis de sentimientos y visualización
    """
    print("=== EJEMPLO DE GRÁFICO DE RADAR DE SENTIMIENTOS ===")
    
    # 1. Crear datos de ejemplo
    print("1. Creando datos de ejemplo...")
    df = create_sample_data()
    print(f"   ✓ {len(df)} registros creados")
    
    # 2. Procesar análisis de sentimientos
    print("2. Procesando análisis de sentimientos...")
    df_processed = process_text_features(df, text_column='transcript_clean')
    
    # 3. Mostrar columnas disponibles
    sentiment_cols = [col for col in df_processed.columns if col.startswith('sentiment_')]
    print(f"   ✓ Columnas de sentimientos creadas: {len(sentiment_cols)}")
    for col in sentiment_cols[:5]:  # Mostrar las primeras 5
        print(f"     - {col}")
    if len(sentiment_cols) > 5:
        print(f"     ... y {len(sentiment_cols) - 5} más")
    
    # 4. Crear gráfico de radar por años
    print("3. Creando gráfico de radar por años...")
    create_sentiment_radar_chart(df_processed, year_column='year', group_by_years=True)
    
    # 5. Crear gráfico de radar agregado
    print("4. Creando gráfico de radar agregado...")
    create_sentiment_radar_chart(df_processed, group_by_years=False)
    
    print("✓ Ejemplo completado!")
    return df_processed

def run_with_your_data(df, text_column='transcript_clean', year_column='year'):
    """
    Función para usar con tus propios datos
    
    Args:
        df: Tu DataFrame con datos
        text_column: Nombre de la columna con texto para analizar
        year_column: Nombre de la columna con años (opcional)
    """
    print("=== ANÁLISIS DE SENTIMIENTOS CON TUS DATOS ===")
    
    # Verificar si ya tiene análisis de sentimientos
    sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
    
    if not sentiment_cols:
        print("Procesando análisis de sentimientos...")
        df = process_text_features(df, text_column=text_column)
    else:
        print(f"✓ Ya hay {len(sentiment_cols)} columnas de sentimientos")
    
    # Crear gráficos de radar
    if year_column in df.columns:
        print("Creando gráfico de radar por años...")
        create_sentiment_radar_chart(df, year_column=year_column, group_by_years=True)
    
    print("Creando gráfico de radar agregado...")
    create_sentiment_radar_chart(df, group_by_years=False)
    
    return df

if __name__ == "__main__":
    # Ejecutar ejemplo
    sample_df = run_sentiment_analysis_example()
    
    print("\n" + "="*50)
    print("Para usar con tus propios datos:")
    print("from example_radar_chart import run_with_your_data")
    print("df_result = run_with_your_data(your_dataframe)")
