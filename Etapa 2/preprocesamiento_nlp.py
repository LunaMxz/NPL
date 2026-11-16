"""
Preprocesamiento lingüístico (Etapa 2).

Transforma 'comentario_limpio' en 'comentario_procesado':
    1. Tokenización y lematización con spaCy (modelo en_core_web_sm).
    2. Minúsculas.
    3. Solo tokens alfabéticos (descarta puntuación, números y símbolos).
    4. Eliminación de stopwords (lista de spaCy) CONSERVANDO las negaciones
       definidas en config.STOPWORDS_CONSERVAR.

Escalabilidad: los textos idénticos se procesan una sola vez (caché) y el
modelo se aplica por lotes con nlp.pipe (paralelizable con N_PROCESOS_NLP).
"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from config import MODELO_SPACY, STOPWORDS_CONSERVAR, BATCH_SIZE_NLP, N_PROCESOS_NLP

STOPWORDS = frozenset(STOP_WORDS) - STOPWORDS_CONSERVAR


def cargar_modelo(nombre=MODELO_SPACY):
    """Carga spaCy sin analizador sintáctico ni NER (no se necesitan)."""
    try:
        return spacy.load(nombre, disable=["parser", "ner"])
    except OSError as e:
        raise OSError(
            f"Modelo '{nombre}' no instalado. Ejecuta: python -m spacy download {nombre}"
        ) from e


def procesar_doc(doc):
    """Devuelve la cadena de lemas útiles de un documento spaCy."""
    lemas = []
    for tok in doc:
        lema = tok.lemma_.lower()
        if lema in STOPWORDS_CONSERVAR:          # p. ej. "n't" -> "not"
            lemas.append(lema)
            continue
        if not tok.is_alpha:
            continue
        if lema in STOPWORDS or tok.lower_ in STOPWORDS:
            continue
        lemas.append(lema)
    return " ".join(lemas)


def procesar_textos(textos, nlp, batch_size=BATCH_SIZE_NLP, n_process=N_PROCESOS_NLP):
    """Procesa una lista de textos conservando el orden; deduplica para ahorrar cómputo."""
    textos = [str(t) for t in textos]
    unicos = list(dict.fromkeys(textos))
    docs = nlp.pipe(unicos, batch_size=batch_size, n_process=n_process)
    cache = {t: procesar_doc(d) for t, d in zip(unicos, docs)}
    return [cache[t] for t in textos]