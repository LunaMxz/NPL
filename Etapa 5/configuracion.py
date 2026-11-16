"""
Configuración de la Etapa 5 (corpus general de apoyo y transferencia a comentarios de estudiantes).
Todo parámetro ajustable vive aquí.
"""

# ------------------------------------------------------------------ datos
COLUMNAS_CRUDO = ["id", "materia", "sentimiento", "comentario"]
CLASES_VALIDAS = ["positive", "negative", "neutral", "irrelevant"]
ORDEN_CLASES = ["positive", "neutral", "negative", "irrelevant"]
CLASES_TRANSFERENCIA = ["positive", "neutral", "negative"]   # las del corpus de estudiantes
MIN_TOKENS_UTILES = 2                                         # igual que la Etapa 1

# Las 19 materias académicas (las mismas de la Etapa 1): se EXCLUYEN del corpus general.
CATALOGO_MATERIAS = [
    "Agile Project Management", "Big Data Analytics", "Cloud Computing Essentials",
    "Computer Networks 101", "Cybersecurity Introduction", "Data Visualization with Power BI",
    "Database Systems Advanced", "Deep Learning Fundamentals", "Digital Marketing Strategy",
    "Intro to AI Ethics", "Java Programming Zero-to-Hero", "Machine Learning with TensorFlow",
    "Mobile App Development with Flutter", "Natural Language Processing",
    "Python for Data Science", "React Frontend Mastery", "Software Engineering Principles",
    "UI/UX Design Basics", "Web Development Bootcamp",
]

# ----------------------------------------------------- preprocesamiento (Etapa 2)
MODELO_SPACY = "en_core_web_sm"
STOPWORDS_CONSERVAR = frozenset({"no", "not", "never", "nor", "none", "neither",
                                 "nothing", "nobody", "nowhere", "cannot"})
BATCH_SIZE_NLP = 256

# ----------------------------------------------------- vectorización TF-IDF
NGRAMAS = 2
MIN_DF = 5               # un término debe aparecer en >= MIN_DF documentos de entrenamiento

# ------------------------------------------------------------- evaluación
SEMILLA = 42
TEST_SIZE = 0.2
SEMILLAS = (42, 1, 2, 3, 4)       # G1: particiones repetidas
N_BOOTSTRAP = 1000
NIVEL_CONFIANZA = 0.95

# --------------------------------------------------------------- modelos
MAX_ITER = 1000
TOL = 1e-4
C_DEFECTO = 1.0
ALPHA_DEFECTO = 1.0
MAX_TRAIN_LIBSVM = 10000          # libsvm es costoso: se entrena con a lo sumo este número de documentos

# ------------------------------------------------------------- escalamiento (G3)
FRACCIONES = (0.05, 0.1, 0.25, 0.5, 1.0)
REPETICIONES_TIEMPO = 3

# ----------------------------------------------------------- modo rápido
RAPIDO = dict(SEMILLAS=(42,), N_BOOTSTRAP=200, MAX_TRAIN_LIBSVM=3000,
              FRACCIONES=(0.25, 1.0), REPETICIONES_TIEMPO=1)
SUBMUESTRA_RAPIDA = 15000         # en modo rápido el corpus general se recorta a este número de textos