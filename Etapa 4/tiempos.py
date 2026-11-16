"""
Medición de tiempos (E6 y E7), con un protocolo pensado para ser neutral:

  - Calentamiento descartado y muchas repeticiones (los ajustes duran milisegundos).
  - Orden ALEATORIO intercalado entre modelos: evita que la deriva del sistema favorezca a uno.
  - Recolector de basura desactivado durante la medición; hilos de BLAS/OpenMP limitados.
  - Tiempo de CPU (process_time, lo que pide la propuesta) y tiempo de reloj (perf_counter).
  - Entrenamiento e inferencia por separado; la normalización L2 se cuenta dentro del modelo.
  - Resumen con mediana, rango intercuartílico e intervalo bootstrap; tiempos relativos a Naive Bayes.
  - El benchmark solo mide: la convergencia de cada modelo se verifica ANTES, y las métricas se
    calculan en otro lugar, para no contaminar la medición.
"""
import contextlib
import gc
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import vstack

import config as cfg
from modelos import ajustar, normalizar_l2
from vectorizacion import TfidfRepo


def _limitar_hilos(n):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:
        return contextlib.nullcontext()


def _medir(funcion):
    c0, w0 = time.process_time_ns(), time.perf_counter_ns()
    funcion()
    c1, w1 = time.process_time_ns(), time.perf_counter_ns()
    return (c1 - c0) / 1e6, (w1 - w0) / 1e6          # milisegundos: CPU, reloj


def _una_repeticion(espec, x_tr, y_tr, x_te):
    modelo = espec.crear(espec.valor_defecto)

    def entrenar():
        modelo.fit(normalizar_l2(x_tr) if espec.normalizar else x_tr, y_tr)

    def inferir():
        modelo.predict(normalizar_l2(x_te) if espec.normalizar else x_te)

    fit_cpu, fit_pared = _medir(entrenar)
    pred_cpu, pred_pared = _medir(inferir)
    return fit_cpu, fit_pared, pred_cpu, pred_pared


def benchmark(especs, x_tr, y_tr, x_te, repeticiones=None, calentamiento=None, semilla=cfg.SEMILLA):
    repeticiones = repeticiones or cfg.REPETICIONES_TIEMPO
    calentamiento = cfg.CALENTAMIENTO if calentamiento is None else calentamiento
    for e in especs:                                   # la convergencia se verifica antes de medir
        ajustar(e, x_tr, y_tr)
    tareas = [e for e in especs for _ in range(repeticiones)]
    np.random.RandomState(semilla).shuffle(tareas)     # orden aleatorio intercalado
    filas = []
    with warnings.catch_warnings(), _limitar_hilos(cfg.HILOS):
        warnings.simplefilter("ignore")
        for e in especs:
            for _ in range(calentamiento):
                _una_repeticion(e, x_tr, y_tr, x_te)
        gc.collect()
        gc.disable()
        try:
            for i, e in enumerate(tareas):
                fc, fp, pc, pp = _una_repeticion(e, x_tr, y_tr, x_te)
                filas.append((e.nombre, i, fc, fp, pc, pp))
        finally:
            gc.enable()
    return pd.DataFrame(filas, columns=["modelo", "orden", "fit_cpu_ms", "fit_reloj_ms",
                                        "pred_cpu_ms", "pred_reloj_ms"])


def benchmark_vectorizacion(textos_tr, textos_te, repeticiones=None, calentamiento=None):
    repeticiones = repeticiones or cfg.REPETICIONES_TIEMPO
    calentamiento = cfg.CALENTAMIENTO if calentamiento is None else calentamiento
    filas = []
    with _limitar_hilos(cfg.HILOS):
        for _ in range(calentamiento):
            TfidfRepo().fit_transform(textos_tr)
        gc.collect()
        gc.disable()
        try:
            for i in range(repeticiones):
                v = TfidfRepo()
                fc, fp = _medir(lambda: v.fit_transform(textos_tr))
                pc, pp = _medir(lambda: v.transform(textos_te))
                filas.append(("TF-IDF (vectorización)", i, fc, fp, pc, pp))
        finally:
            gc.enable()
    return pd.DataFrame(filas, columns=["modelo", "orden", "fit_cpu_ms", "fit_reloj_ms",
                                        "pred_cpu_ms", "pred_reloj_ms"])


def resumir(crudos, referencia="Naive Bayes", n_boot=None, semilla=cfg.SEMILLA):
    """Mediana, IQR, media, sd, CV e IC bootstrap de la mediana; razón de medianas frente a la referencia."""
    n_boot = n_boot or cfg.BOOTSTRAP_TIEMPO
    rng = np.random.RandomState(semilla)
    a = (1 - cfg.NIVEL_CONFIANZA) / 2 * 100
    medidas = {"entrenamiento_cpu": "fit_cpu_ms", "entrenamiento_reloj": "fit_reloj_ms",
               "inferencia_cpu": "pred_cpu_ms", "inferencia_reloj": "pred_reloj_ms"}
    filas = []
    for nombre_medida, col in medidas.items():
        ref = crudos.loc[crudos["modelo"] == referencia, col].to_numpy()
        for modelo, g in crudos.groupby("modelo", sort=False):
            v = g[col].to_numpy()
            meds = np.median(rng.choice(v, size=(n_boot, len(v))), axis=1)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            fila = {"modelo": modelo, "medida": nombre_medida, "n": len(v), "mediana_ms": med,
                    "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1, "media_ms": v.mean(),
                    "sd_ms": v.std(ddof=1), "cv_pct": v.std(ddof=1) / v.mean() * 100,
                    "ic_inf_ms": np.percentile(meds, a), "ic_sup_ms": np.percentile(meds, 100 - a)}
            if len(ref) and modelo != "TF-IDF (vectorización)":
                razones = meds / np.median(rng.choice(ref, size=(n_boot, len(ref))), axis=1)
                fila.update(relativo_nb=med / np.median(ref), rel_inf=np.percentile(razones, a),
                            rel_sup=np.percentile(razones, 100 - a))
            filas.append(fila)
    return pd.DataFrame(filas)


def escalamiento(especs, x_tr, y_tr, factores, fracciones, repeticiones=None, semilla=cfg.SEMILLA):
    """Tiempo de entrenamiento (CPU) frente al tamaño del entrenamiento.
    'replicacion': el entrenamiento se repite k veces (datos duplicados; solo orientativo).
    'submuestreo': fracciones estratificadas del entrenamiento (real).
    alpha = pendiente log-log: t ∝ n^alpha."""
    repeticiones = repeticiones or cfg.REPETICIONES_ESCALAMIENTO
    rng = np.random.RandomState(semilla)
    conjuntos = []
    for k in factores:
        conjuntos.append(("replicacion", vstack([x_tr] * k, format="csr"), np.tile(y_tr, k)))
    for f in fracciones:
        sel = np.concatenate([rng.permutation(np.where(y_tr == c)[0])[:max(1, int(np.ceil(f * (y_tr == c).sum())))]
                              for c in np.unique(y_tr)])
        conjuntos.append(("submuestreo", x_tr[np.sort(sel)], y_tr[np.sort(sel)]))
    filas = []
    with warnings.catch_warnings(), _limitar_hilos(cfg.HILOS):
        warnings.simplefilter("ignore")
        for tipo, x, y in conjuntos:
            for e in especs:
                _, info = ajustar(e, x, y, estricto=False)
                tiempos = []
                for _ in range(repeticiones):
                    modelo = e.crear(e.valor_defecto)
                    xe = normalizar_l2(x) if e.normalizar else x
                    c0 = time.process_time_ns()
                    modelo.fit(xe, y)
                    tiempos.append((time.process_time_ns() - c0) / 1e6)
                filas.append({"tipo": tipo, "modelo": e.nombre, "n": x.shape[0],
                              "mediana_ms": float(np.median(tiempos)), "convergio": info["convergio"],
                              "orientativo": tipo == "replicacion"})
    tabla = pd.DataFrame(filas)
    alfas = {}
    for (tipo, modelo), g in tabla.groupby(["tipo", "modelo"]):
        g = g.drop_duplicates("n")
        alfas[(tipo, modelo)] = (np.polyfit(np.log(g["n"]), np.log(g["mediana_ms"]), 1)[0]
                                 if g["n"].nunique() >= 2 else np.nan)
    tabla["alpha"] = [alfas[(t, m)] for t, m in zip(tabla["tipo"], tabla["modelo"])]
    return tabla