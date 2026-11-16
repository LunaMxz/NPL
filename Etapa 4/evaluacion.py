"""
Métricas, referencias (baselines) e intervalos de confianza por bootstrap de textos.

Métricas (propuesta): Accuracy, Precision, Recall y Macro F1, calculadas por comentario.
Además, exactitud por texto: cada texto distinto pesa igual (promedio de la exactitud de cada texto).
Las clases promediadas en las métricas macro son las presentes en y_true o en y_pred (criterio de
scikit-learn por defecto).

Los comentarios con el mismo texto no son observaciones independientes: la unidad estadística es
el texto. Por eso los intervalos se calculan remuestreando TEXTOS (bootstrap por conglomerados).
"""
import numpy as np

import config as cfg

NOMBRES_METRICAS = ("accuracy", "precision", "recall", "f1", "acc_textos")


def matriz_confusion(y, p, k):
    """Filas = clase real, columnas = clase predicha."""
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def metricas_desde_cm(cm):
    tp = np.diag(cm).astype(float)
    soporte, predichos = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    activas = (soporte + predichos) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predichos > 0, tp / predichos, 0.0)
        rec = np.where(soporte > 0, tp / soporte, 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"accuracy": tp.sum() / max(cm.sum(), 1), "precision": prec[activas].mean(),
            "recall": rec[activas].mean(), "f1": f1[activas].mean()}


def exactitud_por_texto(grupos, correctos):
    """Promedio, sobre los textos presentes, de la exactitud de cada texto."""
    n = np.bincount(grupos)
    ok = np.bincount(grupos, weights=correctos.astype(float))
    return float((ok[n > 0] / n[n > 0]).mean())


def calcular_metricas(y, p, grupos, k):
    m = metricas_desde_cm(matriz_confusion(y, p, k))
    m["acc_textos"] = exactitud_por_texto(grupos, y == p)
    return m


# ------------------------------------------------------------------ referencias
def ref_azar(n, k, rng):
    return rng.randint(0, k, n)


def ref_mayoritaria(y_tr, n, k):
    return np.full(n, np.bincount(y_tr, minlength=k).argmax())


def ref_busqueda_exacta(textos_tr, y_tr, textos_te, k):
    """Memoriza texto -> clase más frecuente del entrenamiento. Un texto nunca visto recibe la clase
    mayoritaria. Muestra con un solo número cuánto de la exactitud se logra solo memorizando."""
    por_texto = {}
    for t, c in zip(textos_tr, y_tr):
        por_texto.setdefault(t, np.zeros(k, dtype=int))[c] += 1
    mayoritaria = np.bincount(y_tr, minlength=k).argmax()
    return np.array([por_texto[t].argmax() if t in por_texto else mayoritaria for t in textos_te])


# ---------------------------------------------------------------------- bootstrap
def ic_bootstrap_textos(y, p, grupos, k, n_boot=None, nivel=None, semilla=None):
    """Intervalo percentil por remuestreo de textos (con todas sus predicciones). Devuelve
    {metrica: (inferior, superior)}. Admite varias repeticiones apiladas (el texto conserva su id)."""
    n_boot = n_boot or cfg.N_BOOTSTRAP
    nivel = nivel or cfg.NIVEL_CONFIANZA
    rng = np.random.RandomState(cfg.SEMILLA if semilla is None else semilla)
    ids = np.unique(grupos)
    indices = [np.where(grupos == g)[0] for g in ids]
    acc_texto = np.array([np.mean(y[i] == p[i]) for i in indices])
    muestras = {m: np.empty(n_boot) for m in NOMBRES_METRICAS}
    for b in range(n_boot):
        sel = rng.randint(0, len(ids), len(ids))
        idx = np.concatenate([indices[i] for i in sel])
        m = metricas_desde_cm(matriz_confusion(y[idx], p[idx], k))
        for nombre in ("accuracy", "precision", "recall", "f1"):
            muestras[nombre][b] = m[nombre]
        muestras["acc_textos"][b] = acc_texto[sel].mean()
    a = (1 - nivel) / 2 * 100
    return {m: (float(np.percentile(v, a)), float(np.percentile(v, 100 - a))) for m, v in muestras.items()}