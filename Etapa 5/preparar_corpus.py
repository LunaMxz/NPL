"""
ETAPA 5 - Paso 1: preparar el corpus general de apoyo.

Parte del archivo crudo (el mismo Feedback_Training.csv de la Etapa 1) y conserva las filas que la
Etapa 1 excluyó por no ser materias académicas. Aplica la misma limpieza (Etapa 1) y el mismo
preprocesamiento lingüístico (Etapa 2), y deduplica por texto procesado.

Uso:
    python preparar_corpus.py --entrada "../Etapa 1/Feedback_Training.csv"
    python preparar_corpus.py --entrada "..." --rapido      (recorta el corpus, solo para probar)

Salida (en --salida, por defecto ./salida_etapa5/corpus):
    corpus_general.csv     texto distinto por fila: id, materia, sentimiento, comentario,
                           comentario_limpio, comentario_procesado
    estadisticas_corpus_general.json
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import pandas as pd

import configuracion as cfg

RE_URL = re.compile(r"(https?://\S+|pic\.\S+\.\S+(?:/\S+)?)")
RE_MENCION_ROTA = re.compile(r"@\s+")
RE_ESPACIOS = re.compile(r"\s+")
RE_PUNTUACION_REPETIDA = re.compile(r"([!?.])\1{2,}")
RE_TOKEN_UNK = re.compile(r"\[UNK\]|<unk>", re.IGNORECASE)


def limpiar_texto(texto):
    """Misma limpieza que limpieza.limpiar_texto de la Etapa 1."""
    t = RE_TOKEN_UNK.sub(" ", str(texto))
    t = RE_URL.sub(" ", t)
    t = RE_MENCION_ROTA.sub("@", t)
    t = RE_PUNTUACION_REPETIDA.sub(r"\1\1\1", t)
    return RE_ESPACIOS.sub(" ", t).strip()


def leer_crudo(ruta):
    """Lee el CSV sin encabezado; separa las filas con número de columnas distinto de 4."""
    for enc in ("utf-8", "latin-1"):
        ok, malas = [], 0
        try:
            with open(ruta, encoding=enc, newline="") as f:
                for fila in csv.reader(f):
                    if len(fila) == len(cfg.COLUMNAS_CRUDO):
                        ok.append(fila)
                    elif fila:
                        malas += 1
            return pd.DataFrame(ok, columns=cfg.COLUMNAS_CRUDO), malas, enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("No se pudo leer el archivo con utf-8 ni latin-1.")


def corregir_codificacion(serie):
    try:
        import ftfy
    except ImportError:
        print("  (ftfy no está instalado: se omite la corrección de codificación)")
        return serie
    return serie.map(lambda x: ftfy.fix_text(str(x)))


def cargar_spacy():
    import spacy
    try:
        return spacy.load(cfg.MODELO_SPACY, disable=["parser", "ner"])
    except OSError as e:
        raise OSError(f"Falta el modelo: python -m spacy download {cfg.MODELO_SPACY}") from e


def procesar_textos(textos, nlp):
    """Lematización, minúsculas, solo alfabéticos, sin stopwords (conservando negaciones).
    Igual que preprocesamiento_nlp.py de la Etapa 2; procesa cada texto único una sola vez."""
    from spacy.lang.en.stop_words import STOP_WORDS
    stop = frozenset(STOP_WORDS) - cfg.STOPWORDS_CONSERVAR
    unicos = list(dict.fromkeys(textos))
    cache = {}
    for t, doc in zip(unicos, nlp.pipe(unicos, batch_size=cfg.BATCH_SIZE_NLP)):
        lemas = []
        for tok in doc:
            lema = tok.lemma_.lower()
            if lema in cfg.STOPWORDS_CONSERVAR:
                lemas.append(lema)
            elif tok.is_alpha and lema not in stop and tok.lower_ not in stop:
                lemas.append(lema)
        cache[t] = " ".join(lemas)
    return [cache[t] for t in textos]


def main():
    ap = argparse.ArgumentParser(description="Etapa 5: preparar el corpus general")
    ap.add_argument("--entrada", required=True, help="Feedback_Training.csv (crudo)")
    ap.add_argument("--salida", default="./salida_etapa5")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    dir_corpus = os.path.join(args.salida, "corpus")
    os.makedirs(dir_corpus, exist_ok=True)
    t0 = time.time()
    est = {}

    df, malas, enc = leer_crudo(args.entrada)
    est.update(filas_leidas=len(df), filas_mal_formadas=malas, codificacion=enc)
    print(f"Leídas {len(df)} filas ({malas} mal formadas), codificación {enc}")

    df["materia"] = df["materia"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["sentimiento"] = df["sentimiento"].astype(str).str.strip().str.lower()
    es_academica = df["materia"].isin(cfg.CATALOGO_MATERIAS)
    est["filas_academicas_excluidas"] = int(es_academica.sum())
    df = df.loc[~es_academica & df["sentimiento"].isin(cfg.CLASES_VALIDAS)].copy()
    est["filas_generales"] = len(df)
    est["materias_generales"] = int(df["materia"].nunique())
    print(f"Corpus general (sin materias académicas): {len(df)} filas, {df['materia'].nunique()} entidades")

    df["comentario"] = corregir_codificacion(df["comentario"])
    df["comentario_limpio"] = df["comentario"].map(limpiar_texto)
    n_alfa = df["comentario_limpio"].map(lambda t: len(re.findall(r"[a-zA-Z]{2,}", t)))
    df = df.loc[n_alfa >= cfg.MIN_TOKENS_UTILES].copy()
    est["tras_filtrar_bajo_contenido"] = len(df)

    df = df.drop_duplicates(subset=["comentario_limpio", "sentimiento"]).copy()   # acelera spaCy
    if args.rapido:
        df = df.sample(n=min(cfg.SUBMUESTRA_RAPIDA, len(df)), random_state=cfg.SEMILLA)
    print(f"Procesando {df['comentario_limpio'].nunique()} textos con spaCy ...")
    nlp = cargar_spacy()
    df["comentario_procesado"] = procesar_textos(df["comentario_limpio"].tolist(), nlp)
    df = df.loc[df["comentario_procesado"].str.strip() != ""].copy()
    est["tras_preprocesar"] = len(df)

    # Deduplicación global por texto procesado: el texto es la unidad estadística.
    por_texto = df.groupby("comentario_procesado")["sentimiento"].nunique()
    contradictorios = set(por_texto[por_texto > 1].index)
    est["textos_con_etiquetas_contradictorias_descartados"] = len(contradictorios)
    est["filas_descartadas_por_contradiccion"] = int(df["comentario_procesado"].isin(contradictorios).sum())
    df = df.loc[~df["comentario_procesado"].isin(contradictorios)]
    df = df.drop_duplicates(subset=["comentario_procesado"]).reset_index(drop=True)

    est["textos_distintos_finales"] = len(df)
    est["distribucion_clases"] = {k: int(v) for k, v in df["sentimiento"].value_counts().items()}
    est["longitud_media_tokens_procesados"] = round(float(df["comentario_procesado"].str.split().str.len().mean()), 2)
    est["vocabulario_lemas"] = len(set(t for s in df["comentario_procesado"] for t in s.split()))
    est["segundos"] = round(time.time() - t0, 1)

    cols = ["id", "materia", "sentimiento", "comentario", "comentario_limpio", "comentario_procesado"]
    df[cols].to_csv(os.path.join(dir_corpus, "corpus_general.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(dir_corpus, "estadisticas_corpus_general.json"), "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2, ensure_ascii=False)

    print("\n=== CORPUS GENERAL ===")
    for k, v in est.items():
        print(f"  {k}: {v}")
    print(f"\nGuardado en: {dir_corpus}")


if __name__ == "__main__":
    sys.exit(main())