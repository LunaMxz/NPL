"""
Pruebas de verificación de la Etapa 5. Ejecutar (desde la carpeta de la Etapa 5):

    python pruebas_etapa5.py --general ./salida_etapa5/corpus/corpus_general.csv
"""
import argparse
import math
import sys
import traceback

import numpy as np
import pandas as pd

import configuracion as cfg
import experimentos_general as ex

PRUEBAS = []


def prueba(f):
    PRUEBAS.append(f)
    return f


@prueba
def tfidf_reproduce_el_ejemplo_del_repositorio(ctx):
    v = ex.TfidfRepo(ngramas=1, min_df=1).fit(["The cat sat on my face", "The dog sat on my bed"])
    x = v.transform(["The cat sat on my face"]).toarray()[0]
    esperado = (1 / 6) * math.log10(2)
    assert abs(x[v.indice_["cat"]] - esperado) < 1e-12 and round(x[v.indice_["cat"]], 6) == 0.050172
    assert x[v.indice_["The"]] == 0.0 and x[v.indice_["sat"]] == 0.0
    return "0.050172, igual al ejemplo de mayank408/TFIDF"


@prueba
def min_df_descarta_terminos_raros_sin_cambiar_el_idf(ctx):
    textos = ["alpha beta", "alpha gamma", "alpha beta", "delta"]
    v1 = ex.TfidfRepo(ngramas=1, min_df=1).fit(textos)
    v2 = ex.TfidfRepo(ngramas=1, min_df=2).fit(textos)
    assert v2.vocabulario_ == ["alpha", "beta"]
    for t in v2.vocabulario_:
        assert abs(v1.idf_[v1.indice_[t]] - v2.idf_[v2.indice_[t]]) < 1e-12
    return "el IDF de los términos conservados no cambia"


@prueba
def particion_agrupada_sin_fuga(ctx):
    df = pd.read_csv(ctx["general"], encoding="utf-8-sig")
    y = df["sentimiento"].map({c: i for i, c in enumerate(cfg.ORDEN_CLASES)}).to_numpy()
    ids, textos = df["id"].astype(str).to_numpy(), df["comentario_procesado"].astype(str).to_numpy()
    assert len(set(textos)) == len(textos), "hay textos repetidos en el corpus general"
    for s in (42, 1, 2):
        tr, te = ex.particion_por_id(y, ids, s)
        assert not set(ids[tr]) & set(ids[te]) and not set(textos[tr]) & set(textos[te])
        assert len(tr) + len(te) == len(y) and abs(len(te) / len(y) - 0.2) < 0.03
        a, b = ex.particion_por_id(y, ids, s)
        assert (a == tr).all() and (b == te).all(), "partición no determinista"
    return "ni ids ni textos compartidos; tamaño de prueba cercano al 20 %; determinista"


@prueba
def metricas_coinciden_con_scikit_learn(ctx):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rng = np.random.RandomState(0)
    for _ in range(20):
        y, p = rng.randint(0, 4, 300), rng.randint(0, 4, 300)
        m = ex.calcular(y, p, 4)
        assert abs(m["accuracy"] - accuracy_score(y, p)) < 1e-12
        assert abs(m["f1"] - f1_score(y, p, average="macro")) < 1e-12
        assert abs(m["precision"] - precision_score(y, p, average="macro", zero_division=0)) < 1e-12
        assert abs(m["recall"] - recall_score(y, p, average="macro", zero_division=0)) < 1e-12
    return "Accuracy, Precision, Recall y Macro F1 iguales a scikit-learn"


@prueba
def bootstrap_coherente(ctx):
    cfg_n = cfg.N_BOOTSTRAP
    cfg.N_BOOTSTRAP = 200
    try:
        y = np.random.RandomState(1).randint(0, 3, 200)
        ic = ex.bootstrap(y, y.copy(), 3)
        assert all(lo == 1.0 and hi == 1.0 for lo, hi in ic.values())
        p = np.random.RandomState(2).randint(0, 3, 200)
        ic = ex.bootstrap(y, p, 3, grupos=np.arange(200) % 10)
        assert ic["accuracy"][0] <= ex.calcular(y, p, 3)["accuracy"] <= ic["accuracy"][1]
    finally:
        cfg.N_BOOTSTRAP = cfg_n
    return "un modelo perfecto da IC [1, 1]; el punto cae dentro del IC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", default="./salida_etapa5/corpus/corpus_general.csv")
    args = ap.parse_args()
    ctx, fallos = {"general": args.general}, 0
    for f in PRUEBAS:
        nombre = f.__name__.replace("_", " ")
        try:
            print(f"[OK    ] {nombre}: {f(ctx)}")
        except Exception as e:
            fallos += 1
            print(f"[FALLA ] {nombre}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(PRUEBAS) - fallos} de {len(PRUEBAS)} pruebas correctas." if not fallos else f"\n{fallos} prueba(s) con falla.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()