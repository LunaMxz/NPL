"""
ETAPA 5 - Paso 2: experimentos con el corpus general y transferencia a comentarios de estudiantes.

  G1  Evaluación en el corpus general: particiones 80/20 estratificadas y AGRUPADAS POR id (5 semillas).
      Cada texto aparece una sola vez y las variantes de un mismo tuit (mismo id) no se separan,
      de modo que no hay fuga por duplicados ni por casi-duplicados.
  G2  Transferencia: se entrena con el corpus general (3 clases) y se prueba con los comentarios
      de estudiantes (salida de la Etapa 2). Intervalos por bootstrap de textos (15 textos).
  G3  Escalamiento y costo: tiempo de entrenamiento e inferencia frente al tamaño del entrenamiento,
      y desempeño asociado (para el balance desempeño / costo de la pregunta de investigación).

Uso:
    python experimentos_general.py --academico "../Etapa 2/salida_etapa2/feedback_training_preprocesado.csv"
    python experimentos_general.py --rapido        (prueba de humo)
"""
import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

import configuracion as cfg

COL_TEXTO, COL_CLASE = "comentario_procesado", "sentimiento"


# =========================================================== TF-IDF (fórmulas de la Etapa 3)
def tokens_repo(texto, ngramas=2):
    bow = str(texto).split(" ")
    if ngramas == 2:
        bow = bow + [bow[i] + " " + bow[i + 1] for i in range(len(bow) - 1)]
    return bow


class TfidfRepo:
    """TF = n(t,d)/len(bow); IDF = log10(N/df), solo con el entrenamiento; TF-IDF = TF*IDF.
    Misma fórmula que las Etapas 3 y 4. Única adición: min_df, que descarta términos que aparecen
    en menos de min_df documentos de entrenamiento (necesario para corpus grandes)."""

    def __init__(self, ngramas=None, min_df=None):
        self.ngramas = cfg.NGRAMAS if ngramas is None else ngramas
        self.min_df = cfg.MIN_DF if min_df is None else min_df

    def fit(self, textos):
        bows = [tokens_repo(t, self.ngramas) for t in textos]
        if not bows:
            raise ValueError("No hay documentos para ajustar el TF-IDF.")
        df = Counter()
        for b in bows:
            df.update(set(b))
        self.vocabulario_ = sorted(t for t, c in df.items() if c >= self.min_df)
        self.indice_ = {t: i for i, t in enumerate(self.vocabulario_)}
        self.n_documentos_ = len(bows)
        self.idf_ = np.array([math.log10(self.n_documentos_ / df[t]) for t in self.vocabulario_], dtype=float)
        return self

    def transform(self, textos):
        indptr, indices, datos = [0], [], []
        for texto in textos:
            bow = tokens_repo(texto, self.ngramas)
            n_bow = float(len(bow))
            for termino, n in Counter(t for t in bow if t in self.indice_).items():
                j = self.indice_[termino]
                indices.append(j)
                datos.append(n / n_bow * self.idf_[j])
            indptr.append(len(indices))
        return csr_matrix((np.asarray(datos, dtype=float), np.asarray(indices, dtype=np.int64),
                           np.asarray(indptr, dtype=np.int64)),
                          shape=(len(indptr) - 1, len(self.vocabulario_)))

    def fit_transform(self, textos):
        return self.fit(textos).transform(textos)


def _nulos(x):
    return np.asarray(abs(x).sum(axis=1)).ravel() == 0


# ===================================================================== modelos
def _especificaciones():
    return [
        {"nombre": "Naive Bayes", "crear": lambda: MultinomialNB(alpha=cfg.ALPHA_DEFECTO), "l2": False, "cap": None},
        {"nombre": "SVM lineal (liblinear)", "l2": True, "cap": None,
         "crear": lambda: LinearSVC(C=cfg.C_DEFECTO, max_iter=cfg.MAX_ITER, tol=cfg.TOL, dual="auto",
                                    random_state=cfg.SEMILLA)},
        {"nombre": "SVM lineal (libsvm)", "l2": True, "cap": cfg.MAX_TRAIN_LIBSVM,
         "crear": lambda: SVC(kernel="linear", C=cfg.C_DEFECTO, tol=cfg.TOL, max_iter=-1)},
        {"nombre": "Regresión Logística", "l2": False, "cap": None,
         "crear": lambda: LogisticRegression(C=cfg.C_DEFECTO, max_iter=cfg.MAX_ITER, tol=cfg.TOL)},
    ]


def submuestra(x, y, cap):
    """Submuestra estratificada reproducible (solo para libsvm, cuyo costo crece muy rápido)."""
    if cap is None or x.shape[0] <= cap:
        return x, y
    idx, _ = train_test_split(np.arange(x.shape[0]), train_size=cap, stratify=y, random_state=cfg.SEMILLA)
    idx = np.sort(idx)
    return x[idx], y[idx]


def ajustar(espec, x_tr, y_tr):
    """Entrena y devuelve (modelo, info) con tiempo de reloj, convergencia y tamaño efectivo."""
    x, y = submuestra(x_tr, y_tr, espec["cap"])
    modelo = espec["crear"]()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        modelo.fit(normalize(x, norm="l2") if espec["l2"] else x, y)
        fit_s = time.perf_counter() - t0
    convergio = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    return modelo, {"fit_s": fit_s, "convergio": convergio, "n_entrenamiento_efectivo": int(x.shape[0])}


def predecir(espec, modelo, x_te):
    t0 = time.perf_counter()
    p = modelo.predict(normalize(x_te, norm="l2") if espec["l2"] else x_te)
    return p, time.perf_counter() - t0


@contextlib.contextmanager
def un_hilo():
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            yield
    except Exception:
        yield


# ==================================================================== métricas
def matriz_confusion(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def metricas_cm(cm):
    tp = np.diag(cm).astype(float)
    soporte, predichos = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    activas = (soporte + predichos) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predichos > 0, tp / predichos, 0.0)
        rec = np.where(soporte > 0, tp / soporte, 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"accuracy": tp.sum() / max(cm.sum(), 1), "precision": prec[activas].mean(),
            "recall": rec[activas].mean(), "f1": f1[activas].mean()}


def calcular(y, p, k):
    return metricas_cm(matriz_confusion(y, p, k))


def bootstrap(y, p, k, grupos=None, semilla=None):
    """IC percentil. Sin grupos: remuestrea comentarios. Con grupos: remuestrea textos completos."""
    rng = np.random.RandomState(cfg.SEMILLA if semilla is None else semilla)
    n_boot, a = cfg.N_BOOTSTRAP, (1 - cfg.NIVEL_CONFIANZA) / 2 * 100
    nombres = ("accuracy", "precision", "recall", "f1")
    muestras = {m: np.empty(n_boot) for m in nombres}
    if grupos is None:
        indices = None
    else:
        ids = np.unique(grupos)
        indices = [np.where(grupos == g)[0] for g in ids]
    for b in range(n_boot):
        if indices is None:
            idx = rng.randint(0, len(y), len(y))
        else:
            idx = np.concatenate([indices[i] for i in rng.randint(0, len(indices), len(indices))])
        m = calcular(y[idx], p[idx], k)
        for nombre in nombres:
            muestras[nombre][b] = m[nombre]
    return {m: (float(np.percentile(v, a)), float(np.percentile(v, 100 - a))) for m, v in muestras.items()}


def referencias(y_tr, y_te, k, semilla):
    rng = np.random.RandomState(semilla)
    return {"Azar": rng.randint(0, k, len(y_te)),
            "Clase mayoritaria": np.full(len(y_te), np.bincount(y_tr, minlength=k).argmax())}


def particion_por_id(y, ids, semilla, n_particiones=5):
    """Partición 80/20 estratificada por clase que NUNCA separa filas con el mismo id.
    Las filas con el mismo id son variantes casi idénticas de un mismo tuit; agruparlas evita que
    casi-duplicados caigan en entrenamiento y prueba (con partición simple, el 54 % de los textos de
    prueba tenía un vecino con similitud coseno >= 0.8 en entrenamiento; agrupando por id, 4.6 %)."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgk = StratifiedGroupKFold(n_splits=n_particiones, shuffle=True, random_state=semilla)
    tr, te = next(sgk.split(np.zeros(len(y)), y, ids))
    return np.sort(tr), np.sort(te)


def _guardar(df, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")


def _cargar_general(ruta, clases):
    df = pd.read_csv(ruta, encoding="utf-8-sig").dropna(subset=[COL_TEXTO, COL_CLASE])
    df = df[df[COL_CLASE].isin(clases)].reset_index(drop=True)
    y = df[COL_CLASE].map({c: i for i, c in enumerate(clases)}).to_numpy()
    return df, df[COL_TEXTO].astype(str).to_numpy(), y


# ======================================================================== G1
def g1_corpus_general(ruta_general, dir_tablas):
    clases = cfg.ORDEN_CLASES
    k = len(clases)
    df, textos, y = _cargar_general(ruta_general, clases)
    ids = df["id"].astype(str).to_numpy()
    assert len(set(textos)) == len(textos), "el corpus general debe tener un texto distinto por fila"
    filas, cms, p_semilla = [], [], {}
    for s in cfg.SEMILLAS:
        tr, te = particion_por_id(y, ids, s)
        assert not set(textos[tr]) & set(textos[te]), "fuga: un texto está en entrenamiento y prueba"
        assert not set(ids[tr]) & set(ids[te]), "fuga: un id está en entrenamiento y prueba"
        t0 = time.perf_counter()
        v = TfidfRepo().fit(textos[tr])
        xtr, xte = v.transform(textos[tr]), v.transform(textos[te])
        t_vec = time.perf_counter() - t0
        nulos = int(_nulos(xte).sum())
        print(f"    semilla {s}: entrenamiento={len(tr)}, prueba={len(te)}, términos={len(v.vocabulario_)}, "
              f"vectores nulos en prueba={nulos}")
        for e in _especificaciones():
            modelo, info = ajustar(e, xtr, y[tr])
            p, pred_s = predecir(e, modelo, xte)
            filas.append({"semilla": s, "modelo": e["nombre"], "tipo": "modelo", **calcular(y[te], p, k),
                          "fit_s": info["fit_s"], "pred_s": pred_s, "vectorizacion_s": t_vec,
                          "convergio": info["convergio"], "n_entrenamiento": len(tr),
                          "n_entrenamiento_efectivo": info["n_entrenamiento_efectivo"], "n_prueba": len(te),
                          "terminos": len(v.vocabulario_), "vectores_nulos_prueba": nulos})
            if s == cfg.SEMILLAS[0]:
                p_semilla[e["nombre"]] = (y[te], p)
                cm = matriz_confusion(y[te], p, k)
                cms += [{"modelo": e["nombre"], "clase_real": clases[i], "clase_predicha": clases[j], "n": int(cm[i, j])}
                        for i in range(k) for j in range(k)]
        for nombre, p in referencias(y[tr], y[te], k, s).items():
            filas.append({"semilla": s, "modelo": nombre, "tipo": "referencia", **calcular(y[te], p, k)})
            if s == cfg.SEMILLAS[0]:
                p_semilla[nombre] = (y[te], p)
    tabla = pd.DataFrame(filas)
    _guardar(tabla, os.path.join(dir_tablas, "G1_semillas.csv"))
    _guardar(pd.DataFrame(cms), os.path.join(dir_tablas, "G1_matrices_confusion.csv"))
    resumen = []
    for nombre, g in tabla.groupby("modelo", sort=False):
        fila = {"modelo": nombre, "tipo": g["tipo"].iloc[0], "n_semillas": len(g)}
        for m in ("accuracy", "precision", "recall", "f1"):
            fila[f"{m}_media"] = g[m].mean()
            fila[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        yy, pp = p_semilla[nombre]
        ic = bootstrap(yy, pp, k)
        for m in ("accuracy", "precision", "recall", "f1"):
            fila[f"{m}_ic_inf"], fila[f"{m}_ic_sup"] = ic[m]
        if g["tipo"].iloc[0] == "modelo":
            fila["fit_s_mediana"], fila["pred_s_mediana"] = g["fit_s"].median(), g["pred_s"].median()
            fila["convergio_todos"] = bool(g["convergio"].all())
        resumen.append(fila)
    resumen = pd.DataFrame(resumen)
    _guardar(resumen, os.path.join(dir_tablas, "G1_resumen.csv"))
    return resumen


# ======================================================================== G2
def g2_transferencia(ruta_general, ruta_academico, dir_tablas):
    clases = cfg.CLASES_TRANSFERENCIA
    k = len(clases)
    _, textos_g, y_g = _cargar_general(ruta_general, clases)
    ac = pd.read_csv(ruta_academico, encoding="utf-8-sig").dropna(subset=[COL_TEXTO, COL_CLASE])
    ac = ac[ac[COL_CLASE].isin(clases)].reset_index(drop=True)
    textos_a = ac[COL_TEXTO].astype(str).to_numpy()
    y_a = ac[COL_CLASE].map({c: i for i, c in enumerate(clases)}).to_numpy()
    nombres = sorted(set(textos_a))
    grupos = np.array([nombres.index(t) for t in textos_a])
    v = TfidfRepo().fit(textos_g)
    xtr, xte = v.transform(textos_g), v.transform(textos_a)
    nulos = _nulos(xte)
    tokens_a = {t for s in nombres for t in s.split()}
    cobertura = len([t for t in tokens_a if t in v.indice_]) / max(len(tokens_a), 1)
    print(f"    entrenamiento general={len(textos_g)} textos; prueba académica={len(ac)} comentarios, "
          f"{len(nombres)} textos distintos; vectores nulos={int(nulos.sum())}; "
          f"cobertura de palabras del vocabulario={cobertura:.2f}")
    filas, cms, preds = [], [], {}
    for e in _especificaciones():
        modelo, info = ajustar(e, xtr, y_g)
        p, _ = predecir(e, modelo, xte)
        preds[e["nombre"]] = p
    for nombre, p in referencias(y_g, y_a, k, cfg.SEMILLA).items():
        preds[nombre] = p
    for nombre, p in preds.items():
        ic = bootstrap(y_a, p, k, grupos=grupos)
        acc_t = float(np.mean([np.mean(p[grupos == g] == y_a[grupos == g]) for g in range(len(nombres))]))
        fila = {"modelo": nombre, "tipo": "referencia" if nombre in ("Azar", "Clase mayoritaria") else "modelo",
                **calcular(y_a, p, k), "acc_textos": acc_t, "n_comentarios": len(y_a), "n_textos": len(nombres),
                "vectores_nulos": int(nulos.sum()), "cobertura_vocabulario": cobertura}
        for m in ("accuracy", "f1"):
            fila[f"{m}_ic_inf"], fila[f"{m}_ic_sup"] = ic[m]
        filas.append(fila)
        cm = matriz_confusion(y_a, p, k)
        cms += [{"modelo": nombre, "clase_real": clases[i], "clase_predicha": clases[j], "n": int(cm[i, j])}
                for i in range(k) for j in range(k)]
    por_texto = []
    for g, texto in enumerate(nombres):
        m = grupos == g
        fila = {"texto": texto, "clase": clases[np.bincount(y_a[m]).argmax()], "n_comentarios": int(m.sum()),
                "vector_nulo": bool(nulos[m].all())}
        for nombre, p in preds.items():
            if nombre in ("Azar", "Clase mayoritaria"):
                continue
            fila[f"{nombre}"] = float((p[m] == y_a[m]).mean())
            fila[f"prediccion: {nombre}"] = clases[np.bincount(p[m], minlength=k).argmax()]
        por_texto.append(fila)
    tabla = pd.DataFrame(filas)
    _guardar(tabla, os.path.join(dir_tablas, "G2_transferencia.csv"))
    _guardar(pd.DataFrame(cms), os.path.join(dir_tablas, "G2_matrices_confusion.csv"))
    _guardar(pd.DataFrame(por_texto), os.path.join(dir_tablas, "G2_por_texto.csv"))
    return tabla


# ======================================================================== G3
def g3_escalamiento(ruta_general, dir_tablas):
    clases = cfg.ORDEN_CLASES
    k = len(clases)
    df, textos, y = _cargar_general(ruta_general, clases)
    ids = df["id"].astype(str).to_numpy()
    tr, te = particion_por_id(y, ids, cfg.SEMILLA)
    filas = []
    with un_hilo():
        for f in cfg.FRACCIONES:
            if f < 1.0:
                sub, _ = train_test_split(tr, train_size=f, random_state=cfg.SEMILLA, stratify=y[tr])
            else:
                sub = tr
            gc.collect()
            t0 = time.perf_counter()
            v = TfidfRepo().fit(textos[sub])
            xtr, xte = v.transform(textos[sub]), v.transform(textos[te])
            t_vec = time.perf_counter() - t0
            print(f"    fracción {f}: n={len(sub)}, términos={len(v.vocabulario_)}, vectorización={t_vec:.1f} s")
            for e in _especificaciones():
                fits, preds, info, p = [], [], None, None
                for _ in range(cfg.REPETICIONES_TIEMPO):
                    modelo, info = ajustar(e, xtr, y[sub])
                    p, t_pred = predecir(e, modelo, xte)
                    fits.append(info["fit_s"])
                    preds.append(t_pred)
                filas.append({"fraccion": f, "modelo": e["nombre"], "n_entrenamiento": len(sub),
                              "n_entrenamiento_efectivo": info["n_entrenamiento_efectivo"],
                              "terminos": len(v.vocabulario_), "vectorizacion_s": t_vec,
                              "fit_s_mediana": float(np.median(fits)), "fit_s_min": float(np.min(fits)),
                              "fit_s_max": float(np.max(fits)), "pred_s_mediana": float(np.median(preds)),
                              "convergio": info["convergio"], "repeticiones": len(fits),
                              **calcular(y[te], p, k)})
    tabla = pd.DataFrame(filas)
    alfas = {}
    for nombre, g in tabla.groupby("modelo"):
        g = g.drop_duplicates("n_entrenamiento_efectivo")
        alfas[nombre] = (np.polyfit(np.log(g["n_entrenamiento_efectivo"]), np.log(g["fit_s_mediana"]), 1)[0]
                         if g["n_entrenamiento_efectivo"].nunique() >= 2 else np.nan)
    tabla["alpha_escalamiento"] = tabla["modelo"].map(alfas)
    _guardar(tabla, os.path.join(dir_tablas, "G3_escalamiento.csv"))
    return tabla


# ============================================================== orquestación
def _huella(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Etapa 5: experimentos G1 a G3")
    ap.add_argument("--general", default="./salida_etapa5/corpus/corpus_general.csv")
    ap.add_argument("--academico", default=None,
                    help="feedback_training_preprocesado.csv de la Etapa 2 (para G2)")
    ap.add_argument("--salida", default="./salida_etapa5")
    ap.add_argument("--experimentos", nargs="+", choices=["G1", "G2", "G3"], default=["G1", "G2", "G3"])
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    if args.rapido:
        for clave, valor in cfg.RAPIDO.items():
            setattr(cfg, clave, valor)
    dir_tablas = os.path.join(args.salida, "tablas")
    duraciones = {}
    print(f"=== ETAPA 5: {'MODO RÁPIDO' if args.rapido else 'EJECUCIÓN COMPLETA'} ===")
    for nombre in args.experimentos:
        t0 = time.perf_counter()
        print(f"  {nombre} ...")
        if nombre == "G1":
            r = g1_corpus_general(args.general, dir_tablas)
            print("\n  Macro F1 en el corpus general (media de semillas; IC 95 % de la semilla 42):")
            for _, x in r.iterrows():
                print(f"    {x['modelo']:28s} {x['f1_media']:.3f}  [{x['f1_ic_inf']:.3f}, {x['f1_ic_sup']:.3f}]")
        elif nombre == "G2":
            if not args.academico:
                print("    (omitido: falta --academico)")
                continue
            r = g2_transferencia(args.general, args.academico, dir_tablas)
            print("\n  Transferencia a comentarios de estudiantes (Macro F1, IC 95 % por bootstrap de textos):")
            for _, x in r.iterrows():
                print(f"    {x['modelo']:28s} {x['f1']:.3f}  [{x['f1_ic_inf']:.3f}, {x['f1_ic_sup']:.3f}]  "
                      f"exactitud por texto {x['acc_textos']:.2f}")
        else:
            g3_escalamiento(args.general, dir_tablas)
        duraciones[nombre] = round(time.perf_counter() - t0, 1)
        print(f"  {nombre}: {duraciones[nombre]} s\n")
    import sklearn
    meta = {"fecha_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "modo": "rapido" if args.rapido else "completo", "comando": " ".join(sys.argv),
            "entrada_general_sha256": _huella(args.general),
            "entrada_academico_sha256": _huella(args.academico) if args.academico else None,
            "duracion_segundos": duraciones,
            "parametros": {k_: v for k_, v in vars(cfg).items() if k_.isupper() and k_ != "RAPIDO"},
            "equipo": {"sistema": platform.platform(), "procesador": platform.processor() or platform.machine(),
                       "nucleos_logicos": os.cpu_count(), "python": sys.version.split()[0],
                       "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "metadatos_etapa5.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f"Tablas en: {dir_tablas}")


if __name__ == "__main__":
    main()