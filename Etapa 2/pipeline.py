"""
ETAPA 2 - Preprocesamiento lingüístico (minúsculas, lematización y stopwords)

Uso:
    python pipeline.py --input ../etapa1_limpieza/salida_pipeline/feedback_training_limpio.csv
    python pipeline.py --input feedback_training_limpio.csv --n-procesos 4   # corpus grande

Salida principal: feedback_training_preprocesado.csv (entrada de la Etapa 3).
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import spacy

import config
from preprocesamiento_nlp import cargar_modelo, procesar_textos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="feedback_training_limpio.csv")
    ap.add_argument("--outdir", default="./salida_etapa2")
    ap.add_argument("--n-procesos", type=int, default=config.N_PROCESOS_NLP)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE_NLP)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if config.COL_ENTRADA not in df.columns:
        raise KeyError(f"Falta la columna '{config.COL_ENTRADA}'. Disponibles: {list(df.columns)}")
    os.makedirs(args.outdir, exist_ok=True)
    n_entrada = len(df)

    nlp = cargar_modelo()
    df[config.COL_SALIDA] = procesar_textos(
        df[config.COL_ENTRADA], nlp, batch_size=args.batch_size, n_process=args.n_procesos)
    df["n_tokens_procesados"] = df[config.COL_SALIDA].str.split().str.len().fillna(0).astype(int)

    sin_contenido = df["n_tokens_procesados"] < config.MIN_TOKENS_PROCESADOS
    n_sin_contenido = int(sin_contenido.sum())
    n_un_termino = int((df["n_tokens_procesados"] == 1).sum())
    df = df.loc[~sin_contenido].copy()

    tokens_antes = df[config.COL_ENTRADA].str.split().str.len().mean()
    tokens_despues = df["n_tokens_procesados"].mean()
    lemas = Counter(t for texto in df[config.COL_SALIDA] for t in texto.split())

    print("=== ETAPA 2: PREPROCESAMIENTO LINGUISTICO ===")
    print(f"Modelo: {config.MODELO_SPACY} (spaCy {spacy.__version__})")
    print(f"Comentarios de entrada: {n_entrada}")
    print(f"Descartados por quedar sin terminos: {n_sin_contenido}")
    print(f"Con un solo termino tras el filtrado: {n_un_termino}")
    print(f"Comentarios de salida: {len(df)}")
    print(f"Longitud promedio (tokens por espacios): {tokens_antes:.2f} -> {tokens_despues:.2f}")
    print(f"Terminos distintos (vocabulario de lemas): {len(lemas)}")
    if "sentimiento" in df.columns:
        print("Distribucion de clases:",
              ", ".join(f"{k} = {v}" for k, v in df['sentimiento'].value_counts().items()))

    df.to_csv(os.path.join(args.outdir, "feedback_training_preprocesado.csv"), index=False)
    pd.DataFrame(lemas.most_common(config.TOP_LEMAS), columns=["lema", "frecuencia"]).to_csv(
        os.path.join(args.outdir, "top_lemas.csv"), index=False)
    with open(os.path.join(args.outdir, "metadatos_etapa2.json"), "w", encoding="utf-8") as f:
        json.dump({
            "python": sys.version.split()[0],
            "spacy": spacy.__version__,
            "modelo_spacy": config.MODELO_SPACY,
            "stopwords_conservadas": sorted(config.STOPWORDS_CONSERVAR),
            "min_tokens_procesados": config.MIN_TOKENS_PROCESADOS,
            "entrada": n_entrada,
            "descartados_sin_terminos": n_sin_contenido,
            "con_un_solo_termino": n_un_termino,
            "salida": int(len(df)),
            "vocabulario_lemas": len(lemas),
        }, f, indent=2, ensure_ascii=False)

    print(f"Archivos guardados en: {args.outdir}")


if __name__ == "__main__":
    main()