"""
Clasificadores (scikit-learn) y normalización L2.

Reglas de la Etapa 4:
  - Misma tolerancia (TOL) y límite alto de iteraciones (MAX_ITER) para todos los modelos.
  - La convergencia es obligatoria: un ajuste que no converge detiene la corrida (estricto=True).
  - La normalización L2 es parte del modelo: se aplica vector por vector, sin aprender nada de
    los datos (no hay fuga de información) y deja intactos los vectores nulos.
"""
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

import config as cfg


@dataclass(frozen=True)
class EspecModelo:
    nombre: str
    corto: str                 # etiqueta breve para las gráficas
    crear: Callable            # crear(valor_del_hiperparametro) -> clasificador
    normalizar: bool           # True: se aplica L2 antes de entrenar y de predecir
    oficial: bool              # False: variante de sensibilidad
    hiperparametro: str        # "alpha" (Naive Bayes) o "C"
    valor_defecto: float


def _nb(a):
    return MultinomialNB(alpha=a)


def _liblinear(c):
    return LinearSVC(C=c, max_iter=cfg.MAX_ITER, tol=cfg.TOL, dual="auto", random_state=cfg.SEMILLA)


def _libsvm(c):
    # libsvm no limita iteraciones: termina por su propio criterio de tolerancia
    return SVC(kernel="linear", C=c, tol=cfg.TOL, max_iter=-1)


def _logistica(c):
    return LogisticRegression(C=c, max_iter=cfg.MAX_ITER, tol=cfg.TOL)


CATALOGO = [
    EspecModelo("Naive Bayes", "NB", _nb, False, True, "alpha", cfg.ALPHA_DEFECTO),
    EspecModelo("SVM lineal (liblinear)", "SVM liblinear", _liblinear, True, True, "C", cfg.C_DEFECTO),
    EspecModelo("SVM lineal (libsvm)", "SVM libsvm", _libsvm, True, True, "C", cfg.C_DEFECTO),
    EspecModelo("Regresión Logística", "Reg. Logística", _logistica, False, True, "C", cfg.C_DEFECTO),
    # --- sensibilidad a la normalización
    EspecModelo("SVM lineal (liblinear), sin normalizar", "SVM liblinear crudo", _liblinear, False, False, "C", cfg.C_DEFECTO),
    EspecModelo("SVM lineal (libsvm), sin normalizar", "SVM libsvm crudo", _libsvm, False, False, "C", cfg.C_DEFECTO),
    EspecModelo("Regresión Logística, con L2", "Reg. Logística L2", _logistica, True, False, "C", cfg.C_DEFECTO),
]


def oficiales():
    return [m for m in CATALOGO if m.oficial]


def por_nombre(nombre):
    return next(m for m in CATALOGO if m.nombre == nombre)


def normalizar_l2(x):
    return normalize(x, norm="l2")


def _iteraciones(modelo):
    n = getattr(modelo, "n_iter_", None)
    return float("nan") if n is None else int(np.max(n))


def ajustar(espec, x_tr, y_tr, valor=None, estricto=True):
    """Entrena y devuelve (modelo, info). info: convergio, iteraciones, vectores_soporte."""
    modelo = espec.crear(espec.valor_defecto if valor is None else valor)
    x = normalizar_l2(x_tr) if espec.normalizar else x_tr
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(x, y_tr)
    convergio = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    if estricto and not convergio:
        raise RuntimeError(f"{espec.nombre} no convergió con max_iter={cfg.MAX_ITER}, tol={cfg.TOL}.")
    soporte = float(modelo.n_support_.sum()) if hasattr(modelo, "n_support_") else float("nan")
    return modelo, {"convergio": convergio, "iteraciones": _iteraciones(modelo), "vectores_soporte": soporte}


def predecir(espec, modelo, x_te):
    return modelo.predict(normalizar_l2(x_te) if espec.normalizar else x_te)


def puntuaciones(espec, modelo, x_te):
    """Puntuaciones por clase (n × k) para las curvas ROC y PR: probabilidades si el modelo las da,
    márgenes de decisión si no (SVM)."""
    x = normalizar_l2(x_te) if espec.normalizar else x_te
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(x)
    return modelo.decision_function(x)