"""
Configuración de la Etapa 4 (modelado y evaluación).
Todo parámetro ajustable vive aquí; ningún otro módulo lleva valores fijos.
"""

# ------------------------------------------------------------------ datos
COL_ID = "id"
COL_MATERIA = "materia"
COL_CLASE = "sentimiento"
COL_TEXTO = "comentario_procesado"        # salida de la Etapa 2; define el "texto distinto"
COL_TEXTO_LIMPIO = "comentario_limpio"    # solo para la sensibilidad con n-gramas de caracteres
ORDEN_CLASES = ["positive", "neutral", "negative", "irrelevant"]

# ----------------------------------------------------- vectorización TF-IDF
NGRAMAS = 2                               # 1 = solo unigramas; 2 = unigramas y bigramas

# ------------------------------------------------------------- particiones
SEMILLA = 42                              # partición oficial y modelos con aleatoriedad
TEST_SIZE = 0.2                           # 80 % entrenamiento / 20 % prueba (propuesta)
N_SEMILLAS = 30                           # E2: particiones estratificadas repetidas
N_PLIEGUES = 5                            # E3: pliegues por grupos de textos
N_REPETICIONES_GRUPOS = 20                # E3: repeticiones con asignaciones aleatorias

# --------------------------------------------------------------- modelos
MAX_ITER = 1000                           # límite alto; la convergencia es obligatoria
TOL = 1e-4                                # tolerancia idéntica para todos los modelos
ALPHA_DEFECTO = 1.0                       # Naive Bayes
C_DEFECTO = 1.0                           # SVM y Regresión Logística
REJILLA_ALPHA = (0.01, 0.1, 0.5, 1.0, 2.0)
REJILLA_C = (0.01, 0.1, 1.0, 10.0, 100.0)

# ---------------------------------------------------------- estadística
N_BOOTSTRAP = 1000                        # remuestreos por textos para los intervalos
NIVEL_CONFIANZA = 0.95

# ------------------------------------------------------------- tiempos
CALENTAMIENTO = 5                         # ejecuciones descartadas por modelo
REPETICIONES_TIEMPO = 100
BOOTSTRAP_TIEMPO = 2000
HILOS = 1                                 # hilos de BLAS/OpenMP durante la medición
FACTORES_ESCALAMIENTO = (1, 2, 4, 8, 16) # E7: replicación del entrenamiento
FRACCIONES_ESCALAMIENTO = (0.1, 0.25, 0.5, 1.0)  # E7: submuestreo real
REPETICIONES_ESCALAMIENTO = 15

# ----------------------------------------------------------- modo rápido
RAPIDO = dict(N_SEMILLAS=5, N_REPETICIONES_GRUPOS=3, N_BOOTSTRAP=200,
              REPETICIONES_TIEMPO=15, CALENTAMIENTO=2, BOOTSTRAP_TIEMPO=300,
              FACTORES_ESCALAMIENTO=(1, 2, 4), FRACCIONES_ESCALAMIENTO=(0.25, 1.0),
              REPETICIONES_ESCALAMIENTO=5)