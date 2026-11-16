"""Utilidades comunes: archivos, huella de datos y datos del equipo."""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd


def guardar_csv(df, ruta):
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")   # utf-8-sig: abre bien en Excel


def leer_csv(ruta):
    return pd.read_csv(ruta, encoding="utf-8-sig")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def guardar_json(obj, ruta):
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def huella_archivo(ruta):
    """SHA-256 del archivo de entrada: permite comprobar que dos corridas usaron los mismos datos."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def info_equipo():
    """Datos necesarios para interpretar los tiempos medidos."""
    import scipy
    import sklearn
    info = {
        "sistema": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }
    try:
        from threadpoolctl import threadpool_info
        info["bibliotecas_numericas"] = [
            {"api": d.get("internal_api"), "hilos": d.get("num_threads")} for d in threadpool_info()]
    except Exception:
        info["bibliotecas_numericas"] = "no disponible"
    return info