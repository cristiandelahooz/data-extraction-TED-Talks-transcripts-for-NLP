"""
Módulo principal que importa y configura todas las funcionalidades
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuración inicial
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Constante para archivo de datos
DEFAULT_DATA_FILE = "ted_talks_en.csv"

# Importar tracker de progreso
try:
    from .progress_tracker import ProgressTracker, real_time_feedback
    print("OK - Sistema de progreso en tiempo real cargado")
except ImportError as e:
    print(f"ERROR importando progress_tracker: {e}")

try:
    from .environment_setup import (
        setup_environment, 
        check_device,
    )
    print("Módulo de configuración del ambiente cargado")
except ImportError as e:
    print(f"Error importando environment_setup: {e}")

try:
    from .data_cleaner import (
        clean_dataset,
        validate_data_quality
    )
    print("Módulo de limpieza de datos cargado")
except ImportError as e:
    print(f"Error importando data_cleaner: {e}")

try:
    from .nlp_processor import (
        load_nlp_models,
        process_text_features,
        create_word_frequency_analysis
    )
    print("Módulo de procesamiento NLP cargado")
except ImportError as e:
    print(f"Error importando nlp_processor: {e}")

try:
    from .visualizer import (
        create_data_overview_plots,
        create_entity_analysis_plots,
        create_interactive_plots,
        print_summary_statistics
    )
    print("Módulo de visualización cargado")
except ImportError as e:
    print(f"Error importando visualizer: {e}")

try:
    from .ml_models import (
        TedTalkClassifier,
        create_ml_pipeline
    )
    print("Módulo de machine learning cargado")
except ImportError as e:
    print(f"Error importando ml_models: {e}")

class TedTalkAnalyzer:
    """
    Clase principal que orquesta todo el análisis de TED Talks
    """
    
    def __init__(self):
        self.data = None
        self.df_clean = None
        self.df_processed = None
        self.nlp_models = None
        self.ml_classifier = None
        self.results = {}
        
    def load_data(self, file_path=DEFAULT_DATA_FILE):
        """Carga el dataset inicial"""
        print(f"\n=== CARGANDO DATASET: {file_path} ===")
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset cargado: {self.data.shape[0]} filas x {self.data.shape[1]} columnas")
            
            # Mostrar información básica
            print("\nColumnas disponibles:")
            for i, col in enumerate(self.data.columns, 1):
                print(f"{i:2d}. {col}")
            
            self.results['data_loaded'] = True
            return self.data
            
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            self.results['data_loaded'] = False
            return None
    
    def setup_environment(self):
        """
        Configura el ambiente y modelos necesarios
        """
        print("\n=== CONFIGURANDO AMBIENTE ===")
        
        try:
            # Configurar ambiente
            setup_environment()
            
            # Verificar dispositivo
            device = check_device()
            
            # Cargar modelos NLP
            self.nlp_models = load_nlp_models()
            
            self.results['environment_setup'] = True
            self.results['device'] = device
            
            print("Ambiente configurado correctamente")
            
        except Exception as e:
            print(f"Error configurando ambiente: {e}")
            self.results['environment_setup'] = False
    
    def clean_data(self):
        """Limpia y prepara los datos"""
        if self.data is None:
            print("Primero debes cargar los datos")
            return None
        
        print("\n=== LIMPIANDO DATOS ===")
        
        try:
            self.df_clean, cleaning_log = clean_dataset(self.data)
            
            # Validar calidad
            quality_results = validate_data_quality(self.df_clean)
            
            self.results['data_cleaning'] = {
                'original_shape': self.data.shape,
                'clean_shape': self.df_clean.shape,
                'cleaning_log': cleaning_log,
                'quality_results': quality_results
            }
            
            print("Datos limpiados correctamente")
            return self.df_clean
            
        except Exception as e:
            print(f"Error limpiando datos: {e}")
            return None
    
    def process_nlp_features(self, text_column='transcript_clean'):
        """Procesa características de NLP"""
        if self.df_clean is None:
            print("Primero debes limpiar los datos")
            return None
        
        print("\n=== PROCESANDO CARACTERÍSTICAS NLP ===")
        
        try:
            # Procesar características de texto
            self.df_processed = process_text_features(
                self.df_clean, 
                text_column=text_column, 
                nlp_models=self.nlp_models
            )
            
            # Análisis de frecuencia de palabras
            word_frequencies = create_word_frequency_analysis(
                self.df_processed, 
                text_column, 
                stop_words=self.nlp_models['stop_words'] if self.nlp_models else None
            )
            
            self.results['nlp_processing'] = {
                'text_column': text_column,
                'word_frequencies': word_frequencies,
                'features_added': [col for col in self.df_processed.columns if 
                                 col.startswith(('sentiment_', 'text_', 'person_', 'org_', 'gpe_'))]
            }
            
            print("✓ Características NLP procesadas correctamente")
            return self.df_processed
            
        except Exception as e:
            print(f"✗ Error procesando NLP: {e.with_traceback()}")
            return None
    
    def create_visualizations(self):
        """Crea todas las visualizaciones"""
        if self.df_clean is None:
            print("Primero debes procesar los datos")
            return
        
        print("\n=== CREANDO VISUALIZACIONES ===")
        
        try:
            # Resumen de estadísticas
            print_summary_statistics(self.df_clean)
            
            # Visualizaciones principales
            create_data_overview_plots(self.df_clean)
            
            # Análisis de entidades
            create_entity_analysis_plots(self.df_processed)
            
            # Gráficos interactivos
            create_interactive_plots(self.df_processed)

            self.results['visualizations'] = True
            print("✓ Visualizaciones creadas correctamente")
            
        except Exception as e:
            print(f"✗ Error creando visualizaciones: {e}")
            self.results['visualizations'] = False
    
    def train_models(self, text_column='transcript_clean', target_column='popularity_numeric'):
        """Entrena modelos de machine learning"""
        if self.df_clean is None:
            print("Primero debes procesar los datos")
            return None
        
        print("\n=== ENTRENANDO MODELOS DE MACHINE LEARNING ===")
        
        try:
            # Crear y ejecutar pipeline de ML
            self.ml_classifier, ml_results = create_ml_pipeline(
                self.df_processed,
                text_column=text_column,
                target_column=target_column
            )
            
            self.results['machine_learning'] = {
                'models_trained': list(self.ml_classifier.models.keys()),
                'evaluation_results': ml_results,
                'best_model': self.ml_classifier.get_best_model()
            }
            
            print("✓ Modelos entrenados correctamente")
            return self.ml_classifier
            
        except Exception as e:
            print(f"✗ Error entrenando modelos: {e}")
            return None
    
    def print_final_summary(self):
        """Imprime un resumen final del análisis"""
        print("\n" + "=" * 60)
        print("RESUMEN FINAL DEL ANÁLISIS")
        print("=" * 60)
        
        if self.results.get('data_loaded'):
            original_shape = self.results['data_cleaning']['original_shape']
            clean_shape = self.results['data_cleaning']['clean_shape']
            print(f"Datos procesados: {original_shape[0]} → {clean_shape[0]} filas")
            print(f"Calidad de datos: {self.results['data_cleaning']['quality_results']['quality_score']:.2f}/10")
        
        if self.results.get('nlp_processing'):
            features_count = len(self.results['nlp_processing']['features_added'])
            print(f"Características NLP creadas: {features_count}")
        
        if self.results.get('machine_learning'):
            models_count = len(self.results['machine_learning']['models_trained'])
            best_model_info = self.results['machine_learning']['best_model']
            print(f"Modelos entrenados: {models_count}")
            if best_model_info[0]:
                print(f"🏆 Mejor modelo: {best_model_info[0]} (F1: {best_model_info[2]:.4f})")
        
        # Estado general
        steps_completed = sum([
            self.results.get('environment_setup', False),
            self.results.get('data_loaded', False),
            'data_cleaning' in self.results,
            'nlp_processing' in self.results,
            self.results.get('visualizations', False),
            'machine_learning' in self.results
        ])
        
        print(f"Pasos completados: {steps_completed}/6")
        
        if steps_completed == 6:
            print("¡Análisis 100% completo!")
        else:
            print("Algunos pasos no se completaron correctamente")