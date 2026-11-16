"""
Experimentos E1 a E9 de la Etapa 4. Cada función escribe sus tablas en dir_tablas y devuelve un DataFrame.

  E1  Partición oficial 80/20 estratificada (semilla 42): modelos oficiales y de sensibilidad
  E2  N_SEMILLAS particiones estratificadas (TF-IDF recalculado en cada una)
  E3  EVALUACIÓN PRIMARIA: pliegues por grupos de textos (repeticiones) y Leave-One-Text-Out
  E4  Partición por texto de la Etapa 3 (diagnóstico)
  E5  Ajuste de hiperparámetros con validación cruzada por grupos, solo dentro del entrenamiento
  E6  Benchmark de tiempos
  E7  Escalamiento del tiempo de entrenamiento
  E8  Sensibilidad a la representación (TF-IDF del repo, de scikit-learn y n-gramas de caracteres)
  E9  Informe de convergencia
"""
import os
import time

import numpy as np
import pandas as pd

import config as cfg
import tiempos
from datos import (complemento, leave_one_text_out, particion_estratificada, particion_por_texto,
                   pliegues_por_grupos, subcorpus)
from evaluacion import (NOMBRES_METRICAS, calcular_metricas, ic_bootstrap_textos, matriz_confusion,
                        ref_azar, ref_busqueda_exacta, ref_mayoritaria)
from modelos import CATALOGO, ajustar, oficiales, predecir, puntuaciones
from util import guardar_csv, guardar_json, info_equipo, leer_csv
from vectorizacion import TfidfRepo

REFERENCIAS = ("Azar", "Clase mayoritaria", "Búsqueda exacta")


# ----------------------------------------------------------- representaciones
def vec_repo(corpus, tr, te):
    v = TfidfRepo().fit(corpus.textos[tr])
    return v.transform(corpus.textos[tr]), v.transform(corpus.textos[te])


def vec_sk_palabras(corpus, tr, te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus.textos[tr])
    return v.transform(corpus.textos[tr]), v.transform(corpus.textos[te])


def vec_sk_caracteres(corpus, tr, te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit(corpus.textos_limpios[tr])
    return v.transform(corpus.textos_limpios[tr]), v.transform(corpus.textos_limpios[te])


REPRESENTACIONES = {
    "TF-IDF del repositorio (palabras 1-2)": vec_repo,
    "TF-IDF de scikit-learn (palabras 1-2)": vec_sk_palabras,
    "TF-IDF de scikit-learn (caracteres 2-5)": vec_sk_caracteres,
}


# ---------------------------------------------------------------- utilidades
def _nulos(x):
    """True en los vectores sin ningún término del vocabulario de entrenamiento."""
    return np.asarray(abs(x).sum(axis=1)).ravel() == 0


def _predicciones_ref(corpus, tr, te, rng):
    k = len(corpus.clases)
    return {"Azar": ref_azar(len(te), k, rng),
            "Clase mayoritaria": ref_mayoritaria(corpus.y[tr], len(te), k),
            "Búsqueda exacta": ref_busqueda_exacta(corpus.textos[tr], corpus.y[tr], corpus.textos[te], k)}


def _fila(nombre, tipo, oficial, met, info=None, **extra):
    fila = {"modelo": nombre, "tipo": tipo, "oficial": oficial, **met}
    fila.update(info or {})
    fila.update(extra)
    return fila


def _filas_cm(nombre, cm, clases):
    return [{"modelo": nombre, "clase_real": clases[i], "clase_predicha": clases[j], "n": int(cm[i, j])}
            for i in range(len(clases)) for j in range(len(clases))]


def _guardar_scores(dir_tablas, nombre_archivo, y, nombres, matrices, clases):
    np.savez_compressed(os.path.join(dir_tablas, nombre_archivo + ".npz"), y=y,
                        **{f"s{i}": m for i, m in enumerate(matrices)})
    guardar_json({"modelos": nombres, "clases": clases}, os.path.join(dir_tablas, nombre_archivo + "_meta.json"))


# ------------------------------------------------------------------------- E1
def e1_particion_oficial(corpus, dir_tablas):
    tr, te = particion_estratificada(corpus)
    xtr, xte = vec_repo(corpus, tr, te)
    k, filas, cms, scores, nombres = len(corpus.clases), [], [], [], []
    for e in CATALOGO:
        modelo, info = ajustar(e, xtr, corpus.y[tr])
        p = predecir(e, modelo, xte)
        filas.append(_fila(e.nombre, "modelo", e.oficial, calcular_metricas(corpus.y[te], p, corpus.grupos[te], k),
                           info, normalizado=e.normalizar))
        cms += _filas_cm(e.nombre, matriz_confusion(corpus.y[te], p, k), corpus.clases)
        if e.oficial:
            scores.append(puntuaciones(e, modelo, xte))
            nombres.append(e.nombre)
    for nombre, p in _predicciones_ref(corpus, tr, te, np.random.RandomState(cfg.SEMILLA)).items():
        filas.append(_fila(nombre, "referencia", False, calcular_metricas(corpus.y[te], p, corpus.grupos[te], k)))
        cms += _filas_cm(nombre, matriz_confusion(corpus.y[te], p, k), corpus.clases)
    tabla = pd.DataFrame(filas)
    tabla["n_entrenamiento"], tabla["n_prueba"], tabla["vocabulario"] = len(tr), len(te), xtr.shape[1]
    guardar_csv(tabla, os.path.join(dir_tablas, "E1_metricas.csv"))
    guardar_csv(pd.DataFrame(cms), os.path.join(dir_tablas, "E1_matrices_confusion.csv"))
    _guardar_scores(dir_tablas, "E1_scores", corpus.y[te], nombres, scores, corpus.clases)
    return tabla


# ------------------------------------------------------------------------- E2
def e2_semillas(corpus, dir_tablas):
    k, filas = len(corpus.clases), []
    for s in range(cfg.N_SEMILLAS):
        tr, te = particion_estratificada(corpus, semilla=s)
        xtr, xte = vec_repo(corpus, tr, te)
        for e in CATALOGO:
            modelo, info = ajustar(e, xtr, corpus.y[tr])
            p = predecir(e, modelo, xte)
            filas.append(_fila(e.nombre, "modelo", e.oficial,
                               calcular_metricas(corpus.y[te], p, corpus.grupos[te], k), info, semilla=s))
        for nombre, p in _predicciones_ref(corpus, tr, te, np.random.RandomState(s)).items():
            filas.append(_fila(nombre, "referencia", False,
                               calcular_metricas(corpus.y[te], p, corpus.grupos[te], k), semilla=s))
    tabla = pd.DataFrame(filas)
    guardar_csv(tabla, os.path.join(dir_tablas, "E2_semillas.csv"))
    return tabla


# ------------------------------------------------------------------------- E3
def _cv_agrupado(corpus, pliegues, especs, vec, rng, con_scores=False):
    """Validación cruzada por grupos: devuelve las predicciones de cada comentario cuando estuvo en prueba."""
    n, k = len(corpus), len(corpus.clases)
    preds = {e.nombre: np.full(n, -1) for e in especs}
    refs = {r: np.full(n, -1) for r in REFERENCIAS}
    scores = {e.nombre: np.zeros((n, k)) for e in especs} if con_scores else None
    nulos, cubiertos = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for te in pliegues:
        tr = complemento(n, te)
        if set(corpus.grupos[tr]) & set(corpus.grupos[te]):
            raise RuntimeError("Fuga detectada: un mismo texto está en entrenamiento y en prueba.")
        xtr, xte = vec(corpus, tr, te)
        nulos[te], cubiertos[te] = _nulos(xte), True
        for e in especs:
            modelo, _ = ajustar(e, xtr, corpus.y[tr])
            preds[e.nombre][te] = predecir(e, modelo, xte)
            if con_scores:
                scores[e.nombre][te] = puntuaciones(e, modelo, xte)
        for nombre, p in _predicciones_ref(corpus, tr, te, rng).items():
            refs[nombre][te] = p
    if not cubiertos.all():
        raise RuntimeError("Hay comentarios que nunca estuvieron en un pliegue de prueba.")
    return preds, refs, scores, nulos


def _resumen_ic(corpus, pila, esquema, oficiales_):
    """Media y desviación entre repeticiones, e IC bootstrap por textos con todas las repeticiones apiladas."""
    k, filas = len(corpus.clases), []
    for nombre, lista in pila.items():
        por_rep = pd.DataFrame([calcular_metricas(corpus.y, p, corpus.grupos, k) for p in lista])
        r = len(lista)
        ic = ic_bootstrap_textos(np.tile(corpus.y, r), np.concatenate(lista), np.tile(corpus.grupos, r), k)
        fila = {"modelo": nombre, "tipo": "referencia" if nombre in REFERENCIAS else "modelo",
                "oficial": nombre in oficiales_, "esquema": esquema, "n_repeticiones": r}
        for m in NOMBRES_METRICAS:
            fila[f"{m}_media"] = por_rep[m].mean()
            fila[f"{m}_sd"] = por_rep[m].std(ddof=1) if r > 1 else np.nan
            fila[f"{m}_ic_inf"], fila[f"{m}_ic_sup"] = ic[m]
        filas.append(fila)
    return pd.DataFrame(filas)


def e3_grupos(corpus, dir_tablas):
    k, R = len(corpus.clases), cfg.N_REPETICIONES_GRUPOS
    nombres_of = [e.nombre for e in oficiales()]
    pila, filas = {}, []
    for r in range(R):
        rng = np.random.RandomState(r)
        pliegues = pliegues_por_grupos(corpus, cfg.N_PLIEGUES, semilla=r)
        preds, refs, scores, nulos = _cv_agrupado(corpus, pliegues, CATALOGO, vec_repo, rng, con_scores=(r == 0))
        for nombre, p in {**preds, **refs}.items():
            pila.setdefault(nombre, []).append(p)
            filas.append({"repeticion": r, "modelo": nombre,
                          "tipo": "referencia" if nombre in REFERENCIAS else "modelo",
                          **calcular_metricas(corpus.y, p, corpus.grupos, k), "vectores_nulos": int(nulos.sum())})
        if r == 0:
            _guardar_scores(dir_tablas, "E3_scores_rep0", corpus.y, nombres_of,
                            [scores[n] for n in nombres_of], corpus.clases)
    guardar_csv(pd.DataFrame(filas), os.path.join(dir_tablas, "E3_grupos_repeticiones.csv"))
    resumen = _resumen_ic(corpus, pila, f"{cfg.N_PLIEGUES} pliegues por grupos × {R}", nombres_of)
    guardar_csv(resumen, os.path.join(dir_tablas, "E3_grupos_resumen.csv"))

    # Leave-One-Text-Out: cada texto distinto completo es el conjunto de prueba una vez
    preds, refs, _, nulos = _cv_agrupado(corpus, leave_one_text_out(corpus), CATALOGO, vec_repo,
                                         np.random.RandomState(cfg.SEMILLA))
    todo = {**preds, **refs}
    loto = _resumen_ic(corpus, {n: [p] for n, p in todo.items()}, "Leave-One-Text-Out", nombres_of)
    guardar_csv(loto, os.path.join(dir_tablas, "E3_loto_resumen.csv"))
    filas = []
    for g, texto in enumerate(corpus.nombres_grupos):
        m = corpus.grupos == g
        fila = {"texto": texto, "clase": corpus.clases[np.bincount(corpus.y[m]).argmax()],
                "n_comentarios": int(m.sum()), "vector_nulo": bool(nulos[m].all())}
        for nombre, p in todo.items():
            fila[nombre] = float((p[m] == corpus.y[m]).mean())
        filas.append(fila)
    guardar_csv(pd.DataFrame(filas), os.path.join(dir_tablas, "E3_loto_por_texto.csv"))
    return resumen


# ------------------------------------------------------------------------- E4
def e4_particion_por_texto(corpus, dir_tablas):
    tr, te = particion_por_texto(corpus)
    xtr, xte = vec_repo(corpus, tr, te)
    k, filas = len(corpus.clases), []
    nulos = int(_nulos(xte).sum())
    for e in CATALOGO:
        modelo, info = ajustar(e, xtr, corpus.y[tr])
        p = predecir(e, modelo, xte)
        filas.append(_fila(e.nombre, "modelo", e.oficial,
                           calcular_metricas(corpus.y[te], p, corpus.grupos[te], k), info))
    for nombre, p in _predicciones_ref(corpus, tr, te, np.random.RandomState(cfg.SEMILLA)).items():
        filas.append(_fila(nombre, "referencia", False, calcular_metricas(corpus.y[te], p, corpus.grupos[te], k)))
    tabla = pd.DataFrame(filas)
    tabla["n_entrenamiento"], tabla["n_prueba"], tabla["vectores_nulos_prueba"] = len(tr), len(te), nulos
    tabla["textos_compartidos"] = len(set(corpus.grupos[tr]) & set(corpus.grupos[te]))
    guardar_csv(tabla, os.path.join(dir_tablas, "E4_particion_por_texto.csv"))
    textos = pd.DataFrame({"texto": corpus.textos[te], "clase": [corpus.clases[i] for i in corpus.y[te]],
                           "vector_nulo": _nulos(xte)}).drop_duplicates("texto")
    guardar_csv(textos, os.path.join(dir_tablas, "E4_textos_prueba.csv"))
    return tabla


# ------------------------------------------------------------------------- E5
def e5_ajuste(corpus, dir_tablas):
    """Hiperparámetros elegidos SOLO con validación cruzada por grupos dentro del entrenamiento."""
    k = len(corpus.clases)
    tr, te = particion_estratificada(corpus)
    sub = subcorpus(corpus, tr)
    pliegues = pliegues_por_grupos(sub, min(cfg.N_PLIEGUES, sub.n_textos_distintos), semilla=cfg.SEMILLA)
    cache = []
    for ite in pliegues:
        itr = complemento(len(sub), ite)
        cache.append((itr, ite) + vec_repo(sub, itr, ite))
    xtr, xte = vec_repo(corpus, tr, te)
    filas, mejores = [], []
    for e in oficiales():
        rejilla = cfg.REJILLA_ALPHA if e.hiperparametro == "alpha" else cfg.REJILLA_C
        for valor in rejilla:
            f1s, convergio = [], True
            for itr, ite, a, b in cache:
                modelo, info = ajustar(e, a, sub.y[itr], valor=valor, estricto=False)
                convergio &= info["convergio"]
                p = predecir(e, modelo, b)
                f1s.append(calcular_metricas(sub.y[ite], p, sub.grupos[ite], k)["f1"])
            filas.append({"modelo": e.nombre, "hiperparametro": e.hiperparametro, "valor": valor,
                          "f1_medio": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else np.nan,
                          "convergio_todos": convergio, "n_pliegues": len(cache)})
        g = pd.DataFrame(filas)
        g = g[(g["modelo"] == e.nombre) & g["convergio_todos"]]
        mejor = float(g.loc[g["f1_medio"].idxmax(), "valor"]) if len(g) else e.valor_defecto  # primer máximo
        f1_cv = float(g["f1_medio"].max()) if len(g) else np.nan
        resultados = {}
        for etiqueta, valor in (("defecto", e.valor_defecto), ("ajustado", mejor)):
            modelo, _ = ajustar(e, xtr, corpus.y[tr], valor=valor)
            resultados[etiqueta] = calcular_metricas(corpus.y[te], predecir(e, modelo, xte), corpus.grupos[te], k)
        mejores.append({"modelo": e.nombre, "hiperparametro": e.hiperparametro, "valor_defecto": e.valor_defecto,
                        "mejor_valor": mejor, "f1_validacion_cruzada": f1_cv,
                        "f1_prueba_defecto": resultados["defecto"]["f1"], "f1_prueba_ajustado": resultados["ajustado"]["f1"],
                        "accuracy_prueba_ajustado": resultados["ajustado"]["accuracy"]})
    guardar_csv(pd.DataFrame(filas), os.path.join(dir_tablas, "E5_ajuste.csv"))
    tabla = pd.DataFrame(mejores)
    guardar_csv(tabla, os.path.join(dir_tablas, "E5_mejores.csv"))
    return tabla


# --------------------------------------------------------------------- E6 y E7
def e6_tiempos(corpus, dir_tablas):
    tr, te = particion_estratificada(corpus)
    xtr, xte = vec_repo(corpus, tr, te)
    crudos = pd.concat([tiempos.benchmark(oficiales(), xtr, corpus.y[tr], xte),
                        tiempos.benchmark_vectorizacion(corpus.textos[tr], corpus.textos[te])],
                       ignore_index=True)
    resumen = tiempos.resumir(crudos)
    guardar_csv(crudos, os.path.join(dir_tablas, "E6_tiempos_crudos.csv"))
    guardar_csv(resumen, os.path.join(dir_tablas, "E6_tiempos_resumen.csv"))
    guardar_json({"equipo": info_equipo(),
                  "protocolo": {"repeticiones": cfg.REPETICIONES_TIEMPO, "calentamiento": cfg.CALENTAMIENTO,
                                "orden": "aleatorio intercalado", "gc": "desactivado durante la medición",
                                "hilos": cfg.HILOS, "medida_principal": "tiempo de CPU (process_time)",
                                "n_entrenamiento": len(tr), "n_prueba": len(te), "terminos": xtr.shape[1]}},
                 os.path.join(dir_tablas, "E6_equipo.json"))
    return resumen


def e7_escalamiento(corpus, dir_tablas):
    tr, te = particion_estratificada(corpus)
    xtr, _ = vec_repo(corpus, tr, te)
    tabla = tiempos.escalamiento(oficiales(), xtr, corpus.y[tr], cfg.FACTORES_ESCALAMIENTO,
                                 cfg.FRACCIONES_ESCALAMIENTO)
    guardar_csv(tabla, os.path.join(dir_tablas, "E7_escalamiento.csv"))
    return tabla


# ------------------------------------------------------------------------- E8
def e8_representaciones(corpus, dir_tablas):
    k, filas = len(corpus.clases), []
    reps = dict(REPRESENTACIONES)
    if corpus.textos_limpios is None:
        reps.pop("TF-IDF de scikit-learn (caracteres 2-5)")
    tr, te = particion_estratificada(corpus)
    for nombre_rep, vec in reps.items():
        xtr, xte = vec(corpus, tr, te)
        for e in oficiales():
            modelo, _ = ajustar(e, xtr, corpus.y[tr])
            met = calcular_metricas(corpus.y[te], predecir(e, modelo, xte), corpus.grupos[te], k)
            filas.append({"representacion": nombre_rep, "esquema": "partición oficial", "modelo": e.nombre,
                          "accuracy_media": met["accuracy"], "f1_media": met["f1"], "f1_sd": np.nan,
                          "vectores_nulos": int(_nulos(xte).sum()), "terminos": xtr.shape[1]})
        acc = {e.nombre: [] for e in oficiales()}
        f1 = {e.nombre: [] for e in oficiales()}
        nulos = []
        for r in range(cfg.N_REPETICIONES_GRUPOS):
            pliegues = pliegues_por_grupos(corpus, cfg.N_PLIEGUES, semilla=r)
            preds, _, _, nul = _cv_agrupado(corpus, pliegues, oficiales(), vec, np.random.RandomState(r))
            nulos.append(int(nul.sum()))
            for nombre, p in preds.items():
                met = calcular_metricas(corpus.y, p, corpus.grupos, k)
                acc[nombre].append(met["accuracy"])
                f1[nombre].append(met["f1"])
        for nombre in acc:
            filas.append({"representacion": nombre_rep, "esquema": "grupos de textos", "modelo": nombre,
                          "accuracy_media": float(np.mean(acc[nombre])), "f1_media": float(np.mean(f1[nombre])),
                          "f1_sd": float(np.std(f1[nombre], ddof=1)) if len(f1[nombre]) > 1 else np.nan,
                          "vectores_nulos": float(np.mean(nulos)), "terminos": np.nan})
    tabla = pd.DataFrame(filas)
    guardar_csv(tabla, os.path.join(dir_tablas, "E8_representaciones.csv"))
    return tabla


# ------------------------------------------------------------------------- E9
def e9_convergencia(dir_tablas, e2=None):
    if e2 is None:
        e2 = leer_csv(os.path.join(dir_tablas, "E2_semillas.csv"))
    m = e2[e2["tipo"] == "modelo"]
    tabla = m.groupby("modelo", sort=False).agg(
        ajustes=("iteraciones", "size"), todos_convergieron=("convergio", "all"),
        iteraciones_media=("iteraciones", "mean"), iteraciones_min=("iteraciones", "min"),
        iteraciones_max=("iteraciones", "max"), vectores_soporte_media=("vectores_soporte", "mean")).reset_index()
    tabla["max_iter_permitido"] = cfg.MAX_ITER
    tabla["tolerancia"] = cfg.TOL
    guardar_csv(tabla, os.path.join(dir_tablas, "E9_convergencia.csv"))
    return tabla


# ------------------------------------------------------------- orquestación
EXPERIMENTOS = {"E1": e1_particion_oficial, "E2": e2_semillas, "E3": e3_grupos, "E4": e4_particion_por_texto,
                "E5": e5_ajuste, "E6": e6_tiempos, "E7": e7_escalamiento, "E8": e8_representaciones}


def ejecutar(corpus, dir_tablas, seleccion=None):
    seleccion = list(seleccion or [*EXPERIMENTOS, "E9"])
    os.makedirs(dir_tablas, exist_ok=True)
    duraciones, e2 = {}, None
    for nombre in seleccion:
        t0 = time.perf_counter()
        print(f"  {nombre} ...", end=" ", flush=True)
        if nombre == "E9":
            e9_convergencia(dir_tablas, e2)
        else:
            res = EXPERIMENTOS[nombre](corpus, dir_tablas)
            if nombre == "E2":
                e2 = res
        duraciones[nombre] = round(time.perf_counter() - t0, 2)
        print(f"{duraciones[nombre]} s")
    return duraciones