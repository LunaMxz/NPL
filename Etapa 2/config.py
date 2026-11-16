"""Configuración de la Etapa 2 (preprocesamiento lingüístico)."""

MODELO_SPACY = "en_core_web_sm"          # corpus en inglés
COL_ENTRADA = "comentario_limpio"        # salida de la Etapa 1
COL_SALIDA = "comentario_procesado"      # entrada de la Etapa 3

# Palabras que NO se eliminan aunque spaCy las clasifique como stopwords:
# las negaciones invierten la polaridad ("not clear" != "clear").
STOPWORDS_CONSERVAR = frozenset({
    "no", "not", "never", "nor", "none", "neither",
    "nothing", "nobody", "nowhere", "cannot",
})

MIN_TOKENS_PROCESADOS = 1    # menos de esto => se descarta el comentario
BATCH_SIZE_NLP = 256
N_PROCESOS_NLP = 1           # >1 paraleliza spaCy en corpus grandes
TOP_LEMAS = 20