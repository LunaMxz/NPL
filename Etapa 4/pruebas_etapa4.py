"""
Pruebas de verificación de la Etapa 4. Ejecutar (desde la carpeta de la Etapa 4):

    python pruebas_etapa4.py --entrada "../Etapa 2/salida_etapa2/feedback_training_preprocesado.csv"

Opcional: --etapa3 "../Etapa 3" (código de referencia) y --etapa3-salida "../Etapa 3/salida_estratificada".
"""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

import config as cfg
from datos import (cargar_corpus, leave_one_text_out, particion_estratificada, particion_por_texto,
                   pliegues_por_grupos)
from evaluacion import (calcular_metricas, ic_bootstrap_textos, matriz_confusion, metricas_desde_cm,
                        ref_busqueda_exacta, ref_mayoritaria)
from modelos import CATALOGO, ajustar, normalizar_l2, por_nombre
from vectorizacion import TfidfRepo

PRUEBAS = []


def prueba(f):
    PRUEBAS.append(f)
    return f


def _cargar_referencia(carpeta_etapa3):
    for c in [carpeta_etapa3, "../Etapa 3", "../etapa3_vectorizacion"]:
        ruta = os.path.join(c, "tfidf_scratch.py") if c else None
        if ruta and os.path.exists(ruta):
            spec = importlib.util.spec_from_file_location("tfidf_scratch_ref", ruta)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


@prueba
def vectorizador_igual_a_la_referencia_de_la_etapa3(ctx):
    ref = _cargar_referencia(ctx["etapa3"])
    if ref is None:
        return "OMITIDA: no se encontró tfidf_scratch.py de la Etapa 3 (usa --etapa3)"
    c = ctx["corpus"]
    tr, te = particion_estratificada(c)
    bows = [ref.tokenizar(t, False, cfg.NGRAMAS) for t in c.textos[tr]]
    voc = ref.construirVocabulario(bows)
    wd = [ref.contarPalabras(b, voc) for b in bows]
    idfs = ref.computeIDF(wd)
    x_ref = pd.DataFrame([ref.computeTFIDF(ref.computeTF(w, b), idfs) for w, b in zip(wd, bows)], columns=voc).to_numpy()
    v = TfidfRepo().fit(c.textos[tr])
    assert v.vocabulario_ == voc, "los vocabularios difieren"
    dif = np.abs(v.transform(c.textos[tr]).toarray() - x_ref).max()
    assert dif < 1e-12, f"diferencia máxima {dif}"
    filas = []
    for t in c.textos[te]:
        b = ref.tokenizar(t, False, cfg.NGRAMAS)
        filas.append(ref.computeTFIDF(ref.computeTF(ref.contarPalabras(b, voc), b), idfs))
    dif_te = np.abs(v.transform(c.textos[te]).toarray() - pd.DataFrame(filas, columns=voc).to_numpy()).max()
    assert dif_te < 1e-12, f"diferencia máxima en prueba {dif_te}"
    return f"diferencia máxima {max(dif, dif_te):.1e}"


@prueba
def coincide_con_los_archivos_de_la_etapa3(ctx):
    carpeta = ctx["etapa3_salida"]
    if not carpeta or not os.path.exists(os.path.join(carpeta, "tfidf_train.csv")):
        return "OMITIDA: no se encontró tfidf_train.csv de la Etapa 3 (usa --etapa3-salida)"
    c = ctx["corpus"]
    tr, te = particion_estratificada(c)
    v = TfidfRepo().fit(c.textos[tr])
    for nombre, idx in (("tfidf_train.csv", tr), ("tfidf_test.csv", te)):
        archivo = pd.read_csv(os.path.join(carpeta, nombre))
        assert list(archivo.columns) == v.vocabulario_, f"columnas distintas en {nombre}"
        dif = np.abs(v.transform(c.textos[idx]).toarray() - archivo.to_numpy()).max()
        assert dif < 1e-12, f"{nombre}: diferencia máxima {dif}"
    return "las matrices de la Etapa 4 y de la Etapa 3 son idénticas"


@prueba
def particiones_sin_fuga_y_completas(ctx):
    c = ctx["corpus"]
    n = len(c)
    tr, te = particion_estratificada(c)
    assert len(tr) + len(te) == n and not set(tr) & set(te)
    tr2, te2 = particion_por_texto(c)
    assert not set(c.grupos[tr2]) & set(c.grupos[te2]), "partición por texto con fuga"
    for semilla in range(3):
        pliegues = pliegues_por_grupos(c, cfg.N_PLIEGUES, semilla)
        todos = np.concatenate(pliegues)
        assert len(todos) == n and len(set(todos)) == n, "un comentario no aparece exactamente una vez"
        textos_por_pliegue = [set(c.grupos[p]) for p in pliegues]
        assert sum(len(t) for t in textos_por_pliegue) == len(set().union(*textos_por_pliegue)), "un texto está en dos pliegues"
        assert all(len(set(c.y[p])) == len(c.clases) for p in pliegues) or c.n_textos_distintos < 3 * cfg.N_PLIEGUES
    assert len(leave_one_text_out(c)) == c.n_textos_distintos
    for f in (particion_estratificada, particion_por_texto):
        assert all((a == b).all() for a, b in zip(f(c), f(c))), "partición no determinista"
    return "sin fugas; cada comentario y cada texto aparecen en un solo pliegue"


@prueba
def normalizacion_l2_conserva_vectores_nulos(ctx):
    from scipy.sparse import csr_matrix
    x = csr_matrix(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    n = normalizar_l2(x).toarray()
    assert np.allclose(np.linalg.norm(n, axis=1), [1, 0, 1]) and not n[1].any()
    return "normas 1 y vectores nulos intactos"


@prueba
def metricas_coinciden_con_scikit_learn(ctx):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rng = np.random.RandomState(0)
    for _ in range(20):
        y, p = rng.randint(0, 3, 200), rng.randint(0, 3, 200)
        m = metricas_desde_cm(matriz_confusion(y, p, 3))
        assert abs(m["accuracy"] - accuracy_score(y, p)) < 1e-12
        assert abs(m["f1"] - f1_score(y, p, average="macro")) < 1e-12
        assert abs(m["precision"] - precision_score(y, p, average="macro", zero_division=0)) < 1e-12
        assert abs(m["recall"] - recall_score(y, p, average="macro", zero_division=0)) < 1e-12
    y = rng.randint(0, 3, 50)
    assert calcular_metricas(y, y, np.arange(50) % 5, 3)["f1"] == 1.0
    return "Accuracy, Precision, Recall y Macro F1 iguales a scikit-learn"


@prueba
def referencias_y_bootstrap(ctx):
    c = ctx["corpus"]
    k = len(c.clases)
    p = ref_busqueda_exacta(c.textos, c.y, c.textos, k)
    assert (p == c.y).all(), "la búsqueda exacta sobre los mismos datos debe acertar todo"
    assert len(set(ref_mayoritaria(c.y, 10, k))) == 1
    ic = ic_bootstrap_textos(c.y, c.y.copy(), c.grupos, k, 200)
    assert all(lo == 1.0 and hi == 1.0 for lo, hi in ic.values()), "IC de un modelo perfecto debe ser [1, 1]"
    rng = np.random.RandomState(1)
    pred = rng.randint(0, k, len(c))
    ic = ic_bootstrap_textos(c.y, pred, c.grupos, k, 300)
    punto = calcular_metricas(c.y, pred, c.grupos, k)
    assert ic["accuracy"][0] <= punto["accuracy"] <= ic["accuracy"][1]
    return "búsqueda exacta = 100 %; IC coherentes"


@prueba
def la_convergencia_es_obligatoria(ctx):
    c = ctx["corpus"]
    tr, _ = particion_estratificada(c)
    x = TfidfRepo().fit_transform(c.textos[tr])
    for e in CATALOGO:
        ajustar(e, x, c.y[tr])                                 # con los parámetros normales, todos convergen
    antes = cfg.MAX_ITER
    cfg.MAX_ITER = 2
    try:
        try:
            ajustar(por_nombre("Regresión Logística"), x, c.y[tr])
        except RuntimeError:
            return "todos los modelos convergen; con max_iter=2 se detiene la corrida"
        raise AssertionError("un ajuste sin convergencia no detuvo la corrida")
    finally:
        cfg.MAX_ITER = antes


@prueba
def la_evaluacion_agrupada_detecta_fugas(ctx):
    import experimentos as ex
    c = ctx["corpus"]
    pliegue_malo = [np.arange(0, len(c), 2)]                  # mezcla textos entre entrenamiento y prueba
    try:
        ex._cv_agrupado(c, pliegue_malo, [por_nombre("Naive Bayes")], ex.vec_repo, np.random.RandomState(0))
    except RuntimeError:
        return "un pliegue con textos compartidos se rechaza"
    raise AssertionError("no se detectó la fuga")


@prueba
def resultados_reproducibles(ctx):
    import experimentos as ex
    c = ctx["corpus"]
    ant = (cfg.N_REPETICIONES_GRUPOS, cfg.N_BOOTSTRAP)
    cfg.N_REPETICIONES_GRUPOS, cfg.N_BOOTSTRAP = 2, 100
    d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        for d in (d1, d2):
            ex.e1_particion_oficial(c, d)
            ex.e3_grupos(c, d)
        for archivo in ("E1_metricas.csv", "E3_grupos_resumen.csv", "E3_loto_por_texto.csv"):
            a, b = pd.read_csv(os.path.join(d1, archivo)), pd.read_csv(os.path.join(d2, archivo))
            pd.testing.assert_frame_equal(a, b)
        return "dos corridas con las mismas semillas dan tablas idénticas"
    finally:
        cfg.N_REPETICIONES_GRUPOS, cfg.N_BOOTSTRAP = ant
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True, help="CSV de la Etapa 2 (feedback_training_preprocesado.csv)")
    ap.add_argument("--etapa3", default="../Etapa 3", help="carpeta con tfidf_scratch.py de la Etapa 3")
    ap.add_argument("--etapa3-salida", default="../Etapa 3/salida_estratificada")
    args = ap.parse_args()
    ctx = {"corpus": cargar_corpus(args.entrada), "etapa3": args.etapa3, "etapa3_salida": args.etapa3_salida}
    fallos = 0
    for f in PRUEBAS:
        nombre = f.__name__.replace("_", " ")
        try:
            detalle = f(ctx)
            estado = "OMITIDA" if str(detalle).startswith("OMITIDA") else "OK     "
            print(f"[{estado}] {nombre}" + (f": {detalle}" if detalle else ""))
        except Exception as e:
            fallos += 1
            print(f"[FALLA ] {nombre}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(PRUEBAS) - fallos} de {len(PRUEBAS)} pruebas correctas." if not fallos else f"\n{fallos} prueba(s) con falla.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()