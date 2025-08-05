"""
Módulo para visualización de datos y resultados
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from math import pi

# Constantes
POPULARITY_CATEGORY_LABEL = 'Categoría de Popularidad'

def setup_plot_style():
    """Configura el estilo de los gráficos"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def create_data_overview_plots(df):
    """
    Crea visualizaciones de resumen del dataset
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Exploratorio del Dataset', fontsize=16, fontweight='bold')

    # 1. Distribución de Views
    axes[0, 0].hist(df['views'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribución de Views')
    axes[0, 0].set_xlabel('Número de Views')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)

    # Estadísticas
    mean_views = df['views'].mean()
    median_views = df['views'].median()
    axes[0, 0].axvline(mean_views, color='red', linestyle='--', label=f'Media: {mean_views:,.0f}')
    axes[0, 0].axvline(median_views, color='green', linestyle='--', label=f'Mediana: {median_views:,.0f}')
    axes[0, 0].legend()

    # 2. Distribución de Categorías de Popularidad
    if 'popularity_category' in df.columns:
        category_counts = df['popularity_category'].value_counts().sort_index()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        bars = axes[0, 1].bar(category_counts.index, category_counts.values, color=colors, alpha=0.8)
        axes[0, 1].set_title('Distribución de Categorías de Popularidad')
        axes[0, 1].set_xlabel('Categoría')
        axes[0, 1].set_ylabel('Número de Videos')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Porcentajes en las barras
        for bar, count in zip(bars, category_counts.values):
            height = bar.get_height()
            percentage = (count / len(df)) * 100
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. Longitud de Transcripciones
    if 'transcript_clean' in df.columns:
        transcript_lengths = df['transcript_clean'].str.len()
        axes[1, 0].hist(transcript_lengths, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Distribución de Longitud de Transcripciones')
        axes[1, 0].set_xlabel('Número de Caracteres')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(True, alpha=0.3)
        
        mean_length = transcript_lengths.mean()
        axes[1, 0].axvline(mean_length, color='red', linestyle='--', 
                          label=f'Media: {mean_length:.0f} caracteres')
        axes[1, 0].legend()

    # 4. Relación Views vs Longitud de Título
    if 'title_clean' in df.columns:
        title_lengths = df['title_clean'].str.len()
        axes[1, 1].scatter(title_lengths, df['views'], alpha=0.6, 
                          color='mediumpurple', s=30)
        axes[1, 1].set_title('Relación: Longitud del Título vs Views')
        axes[1, 1].set_xlabel('Longitud del Título (caracteres)')
        axes[1, 1].set_ylabel('Views')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Línea de tendencia
        z = np.polyfit(title_lengths, df['views'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(title_lengths, p(title_lengths), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.show()

def create_entity_analysis_plots(df):
    """
    Crea visualizaciones para análisis de entidades nombradas
    """
    entity_columns = [col for col in df.columns if col.endswith('_count')]
    
    if not entity_columns:
        print("No se encontraron columnas de entidades para visualizar")
        return
    
    setup_plot_style()
    
    # Resumen de entidades
    entity_totals = {}
    for col in entity_columns:
        entity_type = col.replace('_count', '').upper()
        entity_totals[entity_type] = df[col].sum()
    
    # Boxplot de entidades por categoría de popularidad
    if 'popularity_category' in df.columns and entity_columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribución de Entidades por Categoría de Popularidad', fontsize=16)
        
        # Seleccionar las 4 entidades más comunes
        top_entities = sorted(entity_totals.items(), key=lambda x: x[1], reverse=True)[:4]
        
        for i, (entity_type, _) in enumerate(top_entities):
            row = i // 2
            col = i % 2
            
            entity_col = f'{entity_type.lower()}_count'
            if entity_col in df.columns:
                sns.boxplot(data=df, x='popularity_category', y=entity_col, 
                           ax=axes[row][col], palette='viridis')
                axes[row][col].set_title(f'Distribución de {entity_type} por Popularidad', 
                                       fontsize=12, fontweight='bold')
                axes[row][col].set_xlabel(POPULARITY_CATEGORY_LABEL, fontsize=10)
                axes[row][col].set_ylabel(f'Cantidad de {entity_type}', fontsize=10)
                axes[row][col].grid(True, alpha=0.3)
                axes[row][col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def create_interactive_plots(df):
    """
    Crea gráficos interactivos con Plotly
    """
    # 1. Scatter plot interactivo: Views vs características textuales
    if 'text_word_count' in df.columns:
        fig1 = px.scatter(df, 
                         x='text_word_count', 
                         y='views',
                         color='popularity_category' if 'popularity_category' in df.columns else None,
                         title='Relación entre Cantidad de Palabras y Views',
                         labels={'text_word_count': 'Cantidad de Palabras', 'views': 'Views'},
                         hover_data=['title_clean'] if 'title_clean' in df.columns else None)
        fig1.show()
    
def create_sentiment_radar_chart(df, year_column='year', group_by_years=True):
    """
    Crea un gráfico de radar para análisis de sentimientos por año,
    similar al mostrado en la imagen.
    
    Args:
        df: DataFrame con datos de sentimientos
        year_column: Nombre de la columna que contiene el año
        group_by_years: Si agrupar por años o mostrar datos agregados
    """
    # Verificar que existen las columnas de sentimientos necesarias
    required_sentiment_cols = [
        'sentiment_vader_positive', 'sentiment_vader_negative', 
        'sentiment_vader_neutral', 'sentiment_polarity', 'sentiment_subjectivity'
    ]
    
    available_cols = [col for col in required_sentiment_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️ No se encontraron columnas de sentimientos. Asegúrate de ejecutar el análisis de sentimientos primero.")
        return
    
    # Crear mapeo de emociones basado en las métricas de sentimiento disponibles
    emotion_mapping = {
        'joy': 'sentiment_vader_positive',
        'sadness': 'sentiment_vader_negative', 
        'anger': 'sentiment_vader_negative',  # Usamos negativo como proxy para anger
        'fear': 'sentiment_vader_negative',   # Usamos negativo como proxy para fear
        'surprise': 'sentiment_subjectivity', # Subjetividad puede indicar sorpresa
        'disgust': 'sentiment_vader_negative', # Usamos negativo como proxy para disgust
        'anticipation': 'sentiment_polarity',  # Polaridad positiva puede indicar anticipation
        'trust': 'sentiment_polarity'         # Polaridad positiva puede indicar trust
    }
    
    # Filtrar solo las emociones disponibles
    available_emotions = {k: v for k, v in emotion_mapping.items() if v in df.columns}
    
    if not available_emotions:
        print("⚠️ No se pueden mapear las emociones con las columnas disponibles.")
        return
    
    # Preparar datos
    if group_by_years and year_column in df.columns:
        # Agrupar por años
        years = sorted(df[year_column].unique())
        years = [year for year in years if not pd.isna(year)][:15]  # Máximo 15 años para legibilidad
        
        grouped_data = []
        for year in years:
            year_data = df[df[year_column] == year]
            year_emotions = {}
            
            for emotion, col in available_emotions.items():
                if col in year_data.columns:
                    # Normalizar valores a escala 0-30000 similar a la imagen
                    value = year_data[col].mean()
                    
                    # Transformar valores según el tipo de métrica
                    if col == 'sentiment_polarity':
                        # Polaridad: -1 a 1 -> 0 a 30000
                        normalized_value = ((value + 1) / 2) * 30000
                    elif col == 'sentiment_subjectivity':
                        # Subjetividad: 0 a 1 -> 0 a 30000
                        normalized_value = value * 30000
                    else:
                        # VADER scores: 0 a 1 -> 0 a 30000
                        normalized_value = value * 30000
                    
                    year_emotions[emotion] = max(0, normalized_value)
                else:
                    year_emotions[emotion] = 0
            
            grouped_data.append({
                'year': int(year),
                **year_emotions
            })
        
        # Crear DataFrame con los datos agrupados
        radar_df = pd.DataFrame(grouped_data)
        
    else:
        # Datos agregados sin agrupar por años
        emotion_values = {}
        for emotion, col in available_emotions.items():
            if col in df.columns:
                value = df[col].mean()
                if col == 'sentiment_polarity':
                    normalized_value = ((value + 1) / 2) * 30000
                elif col == 'sentiment_subjectivity':
                    normalized_value = value * 30000
                else:
                    normalized_value = value * 30000
                emotion_values[emotion] = max(0, normalized_value)
            else:
                emotion_values[emotion] = 0
        
        radar_df = pd.DataFrame([{'year': 'All Data', **emotion_values}])
    
    # Crear el gráfico de radar
    emotions = list(available_emotions.keys())
    
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Calcular ángulos para cada emoción
    angles = [n / float(len(emotions)) * 2 * pi for n in range(len(emotions))]
    angles += angles[:1]  # Completar el círculo
    
    # Colores para diferentes años (similar a la imagen)
    colors = plt.cm.Set3(np.linspace(0, 1, len(radar_df)))
    
    # Dibujar cada año/grupo
    for idx, row in radar_df.iterrows():
        values = [row[emotion] for emotion in emotions]
        values += values[:1]  # Completar el círculo
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f"{row['year']}", color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Configurar etiquetas de emociones
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([emotion.title() for emotion in emotions], fontsize=12)
    
    # Configurar escala radial
    ax.set_ylim(0, 30000)
    ax.set_yticks([0, 10000, 20000, 30000])
    ax.set_yticklabels(['0', '10000', '20000', '30000'], fontsize=10)
    ax.grid(True)
    
    # Título y leyenda
    title = 'Análisis de Sentimientos por Año' if group_by_years else 'Análisis de Sentimientos'
    plt.title(title, size=16, fontweight='bold', pad=20)
    
    if len(radar_df) > 1:
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas
    print("\n=== ESTADÍSTICAS DEL RADAR DE SENTIMIENTOS ===")
    for emotion in emotions:
        if group_by_years:
            avg_val = radar_df[emotion].mean()
            max_val = radar_df[emotion].max()
            max_year = radar_df.loc[radar_df[emotion].idxmax(), 'year']
            print(f"{emotion.title()}: Promedio={avg_val:.0f}, Máximo={max_val:.0f} (año {max_year})")
        else:
            val = radar_df[emotion].iloc[0]
            print(f"{emotion.title()}: {val:.0f}")

def print_summary_statistics(df):
    """
    Imprime estadísticas resumidas del dataset
    """
    print("=== ESTADÍSTICAS RESUMIDAS ===")
    print(f"Total de videos: {len(df):,}")
    
    if 'views' in df.columns:
        print(f"Promedio de views: {df['views'].mean():,.0f}")
        print(f"Mediana de views: {df['views'].median():,.0f}")
        print(f"Desviación estándar: {df['views'].std():,.0f}")
    
    # Características textuales
    if 'transcript_clean' in df.columns:
        avg_transcript_length = df['transcript_clean'].str.len().mean()
        print(f"Longitud promedio de transcripción: {avg_transcript_length:.0f} caracteres")
    
    if 'title_clean' in df.columns:
        avg_title_length = df['title_clean'].str.len().mean()
        print(f"Longitud promedio de título: {avg_title_length:.1f} caracteres")
    
    # Sentimientos
    if 'sentiment_polarity' in df.columns:
        avg_polarity = df['sentiment_polarity'].mean()
        print(f"Polaridad promedio de sentimiento: {avg_polarity:.3f}")
    
    # Distribución de categorías
    if 'popularity_category' in df.columns:
        print("\nDistribución de categorías de popularidad:")
        category_dist = df['popularity_category'].value_counts().sort_index()
        for category, count in category_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")