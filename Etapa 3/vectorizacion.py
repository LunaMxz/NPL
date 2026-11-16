"""
Etapa 3 - Vectorización TF-IDF (basada en https://github.com/mayank408/TFIDF).

Uso:
    python vectorizacion.py --entrada datos_limpios.csv
    python vectorizacion.py --entrada datos_limpios.csv --test-size 0.2 --seed 42
    python vectorizacion.py --entrada datos_limpios.csv --ngramas 1

Entrada: columna "comentario_procesado" generada por la Etapa 1.
Por defecto (según la propuesta): partición estratificada 80 % / 20 % y
unigramas + bigramas. Vocabulario e IDF se calculan SOLO con entrenamiento
(evita fuga de información); los documentos de prueba se transforman con
ese vocabulario e IDF. --test-size 0 desactiva la partición.
--particion por-texto: cada texto distinto cae completo en train o en test.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from tfidf_scratch import (
    tokenizar, construirVocabulario, contarPalabras,
    computeTF, computeIDF, computeTFIDF,
)

COL_TEXTO = "comentario_procesado"  # salida de la Etapa 1 (stopwords + lematización)
COL_ETIQUETA = "sentimiento"
COLS_META = ["id", "materia", "sentimiento"]


def particion_por_texto(df, test_size, seed):
    """
    Partición alternativa: cada TEXTO DISTINTO cae completo en train o en test
    (estratificada por clase a nivel de texto). Evita que el mismo texto esté
    en ambos conjuntos. Marca la columna 'particion'.
    """
    rng = np.random.RandomState(seed)
    df["particion"] = "train"
    for clase, grupo in df.groupby(COL_ETIQUETA):
        textos = sorted(grupo[COL_TEXTO].unique())
        rng.shuffle(textos)
        n_test = max(1, int(round(len(textos) * test_size)))
        if n_test >= len(textos):
            raise ValueError(f"Clase '{clase}': muy pocos textos distintos ({len(textos)}) "
                             "para una partición por texto.")
        df.loc[(df[COL_ETIQUETA] == clase) & df[COL_TEXTO].isin(textos[:n_test]),
               "particion"] = "test"
    return df


def vectorizar(textos, wordSet, idfs, normalizar, ngramas):
    """TF-IDF de una lista de textos con vocabulario e IDF ya calculados."""
    filas = []
    for texto in textos:
        bow = tokenizar(texto, normalizar, ngramas)
        if len(bow) == 0:
            raise ValueError(f"Documento vacío tras tokenizar: {texto!r}")
        wordDict = contarPalabras(bow, wordSet)
        tfBow = computeTF(wordDict, bow)
        filas.append(computeTFIDF(tfBow, idfs))
    return pd.DataFrame(filas, columns=wordSet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--particion", choices=["estratificada", "por-texto"],
                    default="estratificada",
                    help="estratificada: propuesta (80/20 por clase). "
                         "por-texto: cada texto distinto cae completo en train o test.")
    ap.add_argument("--ngramas", type=int, choices=[1, 2], default=2)
    # El texto de entrada ya viene normalizado desde la Etapa 1; esta opción
    # solo se activa si se usa una columna sin minúsculas ni puntuación limpia.
    ap.add_argument("--normalizar", action="store_true")
    args = ap.parse_args()
    normalizar = args.normalizar

    df = pd.read_csv(args.entrada, sep=None, engine="python")
    faltan = [c for c in [COL_TEXTO, COL_ETIQUETA] if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas: {faltan}. Disponibles: {list(df.columns)}")
    df = df.reset_index(drop=True)

    # Partición opcional (reproducible)
    if args.test_size > 0 and args.particion == "por-texto":
        df = particion_por_texto(df, args.test_size, args.seed)
    elif args.test_size > 0:
        from sklearn.model_selection import train_test_split
        idx_tr, idx_te = train_test_split(
            df.index, test_size=args.test_size,
            random_state=args.seed, stratify=df[COL_ETIQUETA])
        df["particion"] = "train"
        df.loc[idx_te, "particion"] = "test"
    else:
        df["particion"] = "train"
    tr = df[df["particion"] == "train"]
    te = df[df["particion"] == "test"]
    textos_tr, textos_te = set(tr[COL_TEXTO]), set(te[COL_TEXTO])

    # Pasos del repositorio, aplicados a N documentos
    bows_tr = [tokenizar(t, normalizar, args.ngramas) for t in tr[COL_TEXTO]]
    wordSet = construirVocabulario(bows_tr)
    wordDicts_tr = [contarPalabras(b, wordSet) for b in bows_tr]
    idfs = computeIDF(wordDicts_tr)

    os.makedirs(args.salida, exist_ok=True)
    X_tr = vectorizar(tr[COL_TEXTO], wordSet, idfs, normalizar, args.ngramas)
    X_tr.to_csv(os.path.join(args.salida, "tfidf_train.csv"), index=False)
    if len(te):
        X_te = vectorizar(te[COL_TEXTO], wordSet, idfs, normalizar, args.ngramas)
        X_te.to_csv(os.path.join(args.salida, "tfidf_test.csv"), index=False)

    meta = [c for c in COLS_META if c in df.columns] + ["particion"]
    df[meta].to_csv(os.path.join(args.salida, "etiquetas.csv"), index=False)
    pd.Series(idfs, name="idf").rename_axis("termino").to_csv(
        os.path.join(args.salida, "idf.csv"))
    with open(os.path.join(args.salida, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fuente": "https://github.com/mayank408/TFIDF (TFIDF.ipynb)",
            "TF": "n(t,d) / len(bow_d)",
            "IDF": "log10(N / df(t))",
            "TFIDF": "TF * IDF",
            "normalizar": normalizar,
            "ngramas": args.ngramas,
            "test_size": args.test_size,
            "particion": args.particion if args.test_size > 0 else "ninguna",
            "seed": args.seed,
            "n_train": len(tr), "n_test": len(te),
            "textos_distintos_train": len(textos_tr),
            "textos_distintos_test": len(textos_te),
            "textos_compartidos_train_test": len(textos_tr & textos_te),
            "tamano_vocabulario": len(wordSet),
        }, f, indent=2, ensure_ascii=False)

    print(f"Documentos train={len(tr)} test={len(te)} | vocabulario={len(wordSet)}")
    print(f"Textos distintos: train={len(textos_tr)} test={len(textos_te)} "
          f"| compartidos entre train y test={len(textos_tr & textos_te)}")
    print(f"Salidas en: {args.salida}/")


if __name__ == "__main__":
    main()