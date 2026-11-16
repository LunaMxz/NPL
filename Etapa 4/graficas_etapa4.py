"""
Figuras 4.8 a 4.19 de la Etapa 4, generadas a partir de las tablas guardadas por experimentos.py.
Se puede regenerar sin repetir los experimentos:  python pipeline.py --solo-graficas
"""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

import config as cfg
import estilo as st
from estilo import AZUL, GRIS, GRIS_CLARO, MORADO, NARANJA, ROJO, etiqueta, color_modelo
from util import leer_csv

plt = st.plt
REFS = ("Azar", "Clase mayoritaria", "Búsqueda exacta")
OFICIALES = ["Naive Bayes", "SVM lineal (liblinear)", "SVM lineal (libsvm)", "Regresión Logística"]
AZAR = 1 / 3          # referencia visual con 3 clases; se recalcula con el número real de clases


def _t(dir_tablas, nombre):
    return leer_csv(os.path.join(dir_tablas, nombre))


def _pct(x, _p=None):
    return f"{x * 100:.0f} %"


def _azar(k):
    return 1.0 / k


# ----------------------------------------------------------- 4.8 métricas E1
def fig_4_8(dir_tablas, dir_fig):
    e1 = _t(dir_tablas, "E1_metricas.csv")
    filas = [n for n in [*OFICIALES, "Azar", "Clase mayoritaria", "Búsqueda exacta"] if n in set(e1["modelo"])]
    columnas = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                ("f1", "Macro F1"), ("acc_textos", "Exactitud\npor texto")]
    m = np.array([[e1.loc[e1["modelo"] == f, c].iloc[0] for c, _ in columnas] for f in filas])
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(filas) + 2.2))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels([n for _, n in columnas])
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([etiqueta(f) for f in filas])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(columnas), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(filas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    n_of = len([f for f in filas if f in OFICIALES])
    ax.axhline(n_of - 0.5, color=GRIS, lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    r = e1.iloc[0]
    st.encabezado(ax, "4.8", "Métricas en la partición oficial", f"entrenamiento = {int(r['n_entrenamiento'])}, prueba = {int(r['n_prueba'])}",
                  subir=46)
    st.guardar(fig, dir_fig, "4_8_metricas_particion_oficial")


# ------------------------------------------------------ 4.9 matrices de confusión
def fig_4_9(dir_tablas, dir_fig):
    cm = _t(dir_tablas, "E1_matrices_confusion.csv")
    clases = list(dict.fromkeys(cm["clase_real"]))
    fig, ejes = plt.subplots(2, 2, figsize=(7.6, 7.2))
    for ax, nombre in zip(ejes.ravel(), OFICIALES):
        g = cm[cm["modelo"] == nombre]
        m = np.array([[g[(g.clase_real == a) & (g.clase_predicha == b)]["n"].iloc[0] for b in clases] for a in clases])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1))
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=11,
                        color="white" if m[i, j] > m.max() * 0.55 else "black")
        etiquetas = [st.NOMBRE_CLASE.get(c, c) for c in clases]
        ax.set_xticks(range(len(clases)))
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_yticks(range(len(clases)))
        ax.set_yticklabels(etiquetas, fontsize=9)
        ax.set_title(etiqueta(nombre), fontsize=10.5, fontweight="bold", pad=6)
        ax.tick_params(length=0)
        ax.grid(False)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Clase real")
    for ax in ejes[1, :]:
        ax.set_xlabel("Clase predicha")
    st.encabezado(ejes[0, 0], "4.9", "Matrices de confusión (partición oficial)", subir=38)
    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    st.guardar(fig, dir_fig, "4_9_matrices_confusion")


# ------------------------------------------------------------ 4.10 semillas E2
def fig_4_10(dir_tablas, dir_fig):
    e2 = _t(dir_tablas, "E2_semillas.csv")
    nombres = [n for n in [*OFICIALES, "Azar", "Clase mayoritaria"] if n in set(e2["modelo"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rng = np.random.RandomState(0)
    for i, n in enumerate(nombres):
        v = e2.loc[e2["modelo"] == n, "f1"].to_numpy()
        c = color_modelo(n) if n in OFICIALES else GRIS_CLARO
        ax.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=c, alpha=0.55, edgecolor=GRIS), medianprops=dict(color=GRIS, lw=1.5),
                   whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=14, color=c, edgecolor=GRIS, linewidth=0.4, zorder=3)
        ax.text(i, min(v.max() + 0.045, 1.07), f"μ = {v.mean():.3f}\nσ = {v.std(ddof=1):.3f}", ha="center",
                va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels([etiqueta(n) for n in nombres], fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
    ax.grid(axis="y")
    ax.set_ylabel("Macro F1 en el conjunto de prueba")
    st.encabezado(ax, "4.10", f"Variación entre {int(e2['semilla'].nunique())} particiones estratificadas",
                  f"n = {int(e2['semilla'].nunique())} semillas")
    st.guardar(fig, dir_fig, "4_10_variacion_por_semilla")


# ---------------------------------------------- 4.11 desempeño por grupos E3
def fig_4_11(dir_tablas, dir_fig):
    r = _t(dir_tablas, "E3_grupos_resumen.csv")
    orden = [n for n in [*OFICIALES, "SVM lineal (liblinear), sin normalizar", "SVM lineal (libsvm), sin normalizar",
                         "Regresión Logística, con L2", *REFS] if n in set(r["modelo"])]
    r = r.set_index("modelo").loc[orden]
    k = 3
    fig, ejes = plt.subplots(1, 2, figsize=(9.2, 0.42 * len(orden) + 2.6), sharey=True)
    for ax, (m, titulo) in zip(ejes, [("accuracy", "(a) Accuracy"), ("f1", "(b) Macro F1")]):
        y = np.arange(len(orden))[::-1]
        for yi, n in zip(y, orden):
            v, lo, hi = r.loc[n, f"{m}_media"], r.loc[n, f"{m}_ic_inf"], r.loc[n, f"{m}_ic_sup"]
            sens = n not in OFICIALES and n not in REFS
            ax.barh(yi, v, height=0.62, color=GRIS_CLARO if n in REFS else color_modelo(n),
                    alpha=0.55 if sens else 1.0, hatch="//" if sens else ("..." if n in REFS else None),
                    edgecolor=GRIS, linewidth=0.6, zorder=3)
            ax.errorbar(v, yi, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], color=GRIS, capsize=3, lw=1.1, zorder=4)
            ax.text(min(hi, 1.0) + 0.015, yi, f"{v:.2f}", va="center", fontsize=8.5)
        ax.axvline(_azar(k), color=ROJO, ls="--", lw=1.1, zorder=2)
        ax.set_xlim(0, 1.12)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
        ax.grid(axis="x")
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Valor medio (IC 95 % por bootstrap de textos)")
    ejes[0].set_yticks(np.arange(len(orden))[::-1])
    ejes[0].set_yticklabels([etiqueta(n) for n in orden])
    ejes[0].tick_params(axis="y", length=0)
    st.encabezado(ejes[0], "4.11", "Evaluación primaria: validación por grupos de textos",
                  subir=26)
    st.derecha(ejes[1], r["esquema"].iloc[0], 26)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    fig.legend(handles=[plt.Line2D([0], [0], color=ROJO, ls="--", label=f"Azar ({_azar(k):.2f})"),
                        Patch(facecolor="white", edgecolor=GRIS, hatch="//", label="Variante de sensibilidad"),
                        Patch(facecolor=GRIS_CLARO, edgecolor=GRIS, hatch="...", label="Referencia")],
               loc="lower center", ncol=3, fontsize=9)
    st.guardar(fig, dir_fig, "4_11_desempeno_por_grupos_de_textos")


# ------------------------------------------------------ 4.12 LOTO por texto
def fig_4_12(dir_tablas, dir_fig):
    t = _t(dir_tablas, "E3_loto_por_texto.csv")
    cols = [n for n in [*OFICIALES, "SVM lineal (liblinear), sin normalizar", "SVM lineal (libsvm), sin normalizar",
                        "Regresión Logística, con L2", "Búsqueda exacta"] if n in t.columns]
    orden_clase = {c: i for i, c in enumerate(cfg.ORDEN_CLASES)}
    t = t.assign(_o=t["clase"].map(orden_clase)).sort_values(["_o", "texto"]).reset_index(drop=True)
    m = t[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.4, 0.38 * len(t) + 2.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if m[i, j] > 0.6 else "black")
        ax.text(m.shape[1] - 0.35, i, "sí" if t.loc[i, "vector_nulo"] else "no", ha="center", va="center", fontsize=9,
                fontweight="bold" if t.loc[i, "vector_nulo"] else "normal",
                color=ROJO if t.loc[i, "vector_nulo"] else GRIS)
    ax.text(m.shape[1] - 0.35, -0.95, "Vector\nnulo", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([etiqueta(c).replace(" (", "\n(") for c in cols], fontsize=8.5)
    ax.xaxis.tick_top()
    etiquetas_filas = [f"[{st.NOMBRE_CLASE.get(c, c)[:3]}] {x[:44]}{'…' if len(x) > 44 else ''}"
                       for c, x in zip(t["clase"], t["texto"])]
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(etiquetas_filas, fontsize=8.5)
    for lab, c in zip(ax.get_yticklabels(), t["clase"]):
        lab.set_color({"positive": AZUL, "neutral": "#B9770E", "negative": ROJO}.get(c, GRIS))
    ax.set_xlim(-0.5, m.shape[1] + 0.2)
    ax.tick_params(length=0)
    ax.grid(False)
    st.encabezado(ax, "4.12", "Leave-One-Text-Out: exactitud de cada texto", subir=64)
    st.guardar(fig, dir_fig, "4_12_loto_exactitud_por_texto")


# ------------------------------------------------------------ 4.13 tiempos E6
def fig_4_13(dir_tablas, dir_fig):
    r = _t(dir_tablas, "E6_tiempos_resumen.csv")
    eq = json.load(open(os.path.join(dir_tablas, "E6_equipo.json"), encoding="utf-8"))
    fig, ejes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    for ax, medida, titulo in [(ejes[0], "entrenamiento_cpu", "(a) Entrenamiento"), (ejes[1], "inferencia_cpu", "(b) Inferencia")]:
        g = r[r["medida"] == medida].set_index("modelo")
        nombres = [n for n in [*OFICIALES, "TF-IDF (vectorización)"] if n in g.index]
        if medida == "inferencia_cpu":
            pass
        y = np.arange(len(nombres))[::-1]
        for yi, n in zip(y, nombres):
            v, lo, hi = g.loc[n, "mediana_ms"], g.loc[n, "ic_inf_ms"], g.loc[n, "ic_sup_ms"]
            ax.barh(yi, v, height=0.6, color=color_modelo(n) if n in OFICIALES else GRIS_CLARO,
                    edgecolor=GRIS, linewidth=0.6, zorder=3, hatch=None if n in OFICIALES else "..")
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=GRIS, capsize=3, lw=1.1, zorder=4)
            rel = g.loc[n].get("relativo_nb", np.nan)
            ax.text(hi + g["ic_sup_ms"].max() * 0.02, yi, f"{v:.2f} ms" + (f"  (×{rel:.1f})" if pd.notna(rel) and n != "Naive Bayes" else ""),
                    va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels([etiqueta(n) for n in nombres], fontsize=9)
        ax.set_xlim(0, g["ic_sup_ms"].max() * 1.55)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Tiempo de CPU, mediana e IC 95 % (ms)")
    st.encabezado(ejes[0], "4.13", "Tiempo de CPU por modelo", subir=26)
    st.derecha(ejes[1], f"{eq['protocolo']['repeticiones']} repeticiones, orden aleatorio, {eq['protocolo']['hilos']} hilo", 26)
    fig.tight_layout(w_pad=1.6)
    st.guardar(fig, dir_fig, "4_13_tiempos_cpu")


# -------------------------------------------------- 4.14 exactitud vs tiempo
def fig_4_14(dir_tablas, dir_fig):
    e1, e3, e6 = (_t(dir_tablas, "E1_metricas.csv"), _t(dir_tablas, "E3_grupos_resumen.csv"),
                  _t(dir_tablas, "E6_tiempos_resumen.csv"))
    t = e6[e6["medida"] == "entrenamiento_cpu"].set_index("modelo")
    orden = sorted(OFICIALES, key=lambda n: t.loc[n, "mediana_ms"])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for rango, n in enumerate(orden):
        x = t.loc[n, "mediana_ms"]
        f_of = e1.loc[e1["modelo"] == n, "f1"].iloc[0]
        f_gr, lo, hi = (e3.loc[e3["modelo"] == n, c].iloc[0] for c in ("f1_media", "f1_ic_inf", "f1_ic_sup"))
        c = color_modelo(n)
        ax.plot([x, x], [f_gr, f_of], color=GRIS_CLARO, lw=1.6, zorder=2)
        ax.errorbar(x, f_gr, yerr=[[f_gr - lo], [hi - f_gr]], color=c, capsize=3, lw=1.2, zorder=3)
        ax.scatter(x, f_of, s=75, color=c, edgecolor=GRIS, zorder=4)
        ax.scatter(x, f_gr, s=75, facecolor="white", edgecolor=c, linewidth=2, zorder=4)
        vecino_cerca = rango + 1 < len(orden) and t.loc[orden[rango + 1], "mediana_ms"] / x < 1.08
        derecha_ = not vecino_cerca                    # si el siguiente modelo está muy cerca, la etiqueta va a la izquierda
        ax.annotate(etiqueta(n), (x, f_of), xytext=(9 if derecha_ else -9, 8), textcoords="offset points",
                    ha="left" if derecha_ else "right", fontsize=9.5, fontweight="bold", color=c)
    xs = t.loc[OFICIALES, "mediana_ms"]
    ax.set_xscale("log")
    ax.set_xlim(xs.min() * 0.7, xs.max() * 1.5)
    marcas = [v for v in (0.5, 1, 2, 3, 5, 10, 20, 50, 100) if xs.min() * 0.7 <= v <= xs.max() * 1.5]
    ax.set_xticks(marcas)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.axhline(_azar(3), color=ROJO, ls="--", lw=1.1)
    ax.text(1.008, _azar(3), "azar", transform=ax.get_yaxis_transform(), color=ROJO, va="center", ha="left", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Tiempo de entrenamiento, CPU (ms, escala logarítmica)")
    ax.set_ylabel("Macro F1")
    ax.grid(True)
    st.encabezado(ax, "4.14", "Desempeño frente a costo computacional")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", ms=8, color=GRIS, label="Partición oficial 80/20 (memorización)"),
                        plt.Line2D([0], [0], marker="o", ls="", ms=8, markerfacecolor="white", markeredgecolor=GRIS,
                                   markeredgewidth=2, label="Grupos de textos (generalización, IC 95 %)")],
               loc="lower center", ncol=2, fontsize=9)
    st.guardar(fig, dir_fig, "4_14_desempeno_vs_tiempo")


# ---------------------------------------------------------- 4.15 escalamiento
def fig_4_15(dir_tablas, dir_fig):
    e7 = _t(dir_tablas, "E7_escalamiento.csv")
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 4.5), sharey=True)
    for ax, tipo, titulo in [(ejes[0], "submuestreo", "(a) Submuestreo real del entrenamiento"),
                             (ejes[1], "replicacion", "(b) Entrenamiento replicado ×k (orientativo)")]:
        g = e7[e7["tipo"] == tipo]
        for n in OFICIALES:
            h = g[g["modelo"] == n].sort_values("n")
            if len(h):
                ax.plot(h["n"], h["mediana_ms"], marker="o", ms=5, lw=1.8, color=color_modelo(n),
                        label=f"{etiqueta(n)} (α = {h['alpha'].iloc[0]:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_xlabel("Documentos de entrenamiento (n)")
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=6)
        ax.legend(loc="upper left", fontsize=8.5, title=r"$t \propto n^{\alpha}$", title_fontsize=8.5)
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ejes[0].set_ylabel("Tiempo de entrenamiento, CPU (ms)")
    st.encabezado(ejes[0], "4.15", "Escalamiento del tiempo de entrenamiento", subir=26)
    st.derecha(ejes[1], "α = pendiente log-log", 26)
    fig.tight_layout(w_pad=1.4)
    st.guardar(fig, dir_fig, "4_15_escalamiento")


# ------------------------------------------------------ 4.16 convergencia E2
def fig_4_16(dir_tablas, dir_fig):
    e2 = _t(dir_tablas, "E2_semillas.csv")
    m = e2[(e2["tipo"] == "modelo") & e2["iteraciones"].notna()]
    nombres = list(dict.fromkeys(m["modelo"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.RandomState(0)
    for i, n in enumerate(nombres):
        v = m.loc[m["modelo"] == n, "iteraciones"].to_numpy()
        a1.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=color_modelo(n), alpha=0.55, edgecolor=GRIS), medianprops=dict(color=GRIS),
                   whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
        a1.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=11, color=color_modelo(n), edgecolor=GRIS, linewidth=0.3, zorder=3)
    a1.axhline(cfg.MAX_ITER, color=ROJO, ls="--", lw=1.2)
    a1.text(len(nombres) - 0.55, cfg.MAX_ITER * 1.12, f"límite permitido = {cfg.MAX_ITER}", ha="right", color=ROJO, fontsize=9)
    a1.set_yscale("log")
    a1.set_ylim(3, cfg.MAX_ITER * 2.5)
    a1.set_xticks(range(len(nombres)))
    a1.set_xticklabels([etiqueta(n).replace(" (", "\n(") for n in nombres], fontsize=8)
    a1.set_ylabel("Iteraciones hasta converger")
    a1.grid(axis="y")
    a1.set_title("(a) Iteraciones (todos los ajustes convergieron)", fontsize=10.5, fontweight="bold", pad=6)
    sv = e2[e2["modelo"].str.contains("libsvm") & (e2["tipo"] == "modelo")]
    nombres_sv = list(dict.fromkeys(sv["modelo"]))
    for i, n in enumerate(nombres_sv):
        v = sv.loc[sv["modelo"] == n, "vectores_soporte"].to_numpy()
        a2.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=NARANJA, alpha=0.4 if "sin normalizar" in n else 0.9, edgecolor=GRIS,
                                 hatch="//" if "sin normalizar" in n else None), medianprops=dict(color=GRIS),
                   whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
        a2.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=11, color=NARANJA, edgecolor=GRIS, linewidth=0.3, zorder=3)
    a2.set_xticks(range(len(nombres_sv)))
    a2.set_xticklabels(["con L2" if "sin normalizar" not in n else "sin normalizar" for n in nombres_sv])
    a2.set_ylabel("Vectores de soporte (SVM libsvm)")
    a2.grid(axis="y")
    a2.set_title("(b) Vectores de soporte", fontsize=10.5, fontweight="bold", pad=6)
    st.encabezado(a1, "4.16", "Convergencia de los modelos", subir=26)
    st.derecha(a2, f"tolerancia = {cfg.TOL:g}", 26)
    fig.tight_layout(w_pad=1.6)
    st.guardar(fig, dir_fig, "4_16_convergencia")


# --------------------------------------------------------- 4.17 ajuste E5
def fig_4_17(dir_tablas, dir_fig):
    a, mejores = _t(dir_tablas, "E5_ajuste.csv"), _t(dir_tablas, "E5_mejores.csv").set_index("modelo")
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 4.5), sharey=True)
    for ax, hip, titulo in [(ejes[0], "alpha", "(a) Naive Bayes: suavizado α"), (ejes[1], "C", "(b) Modelos lineales: parámetro C")]:
        g = a[a["hiperparametro"] == hip]
        for n in dict.fromkeys(g["modelo"]):
            h = g[g["modelo"] == n].sort_values("valor")
            c = color_modelo(n)
            ax.errorbar(h["valor"], h["f1_medio"], yerr=h["f1_sd"].fillna(0), marker="o", ms=5, lw=1.8, color=c,
                        capsize=3, label=etiqueta(n))
            b = mejores.loc[n, "mejor_valor"]
            ax.scatter([b], [h.loc[h["valor"] == b, "f1_medio"].iloc[0]], marker="*", s=190, color=c, edgecolor=GRIS, zorder=5)
        ax.axhline(_azar(3), color=ROJO, ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.grid(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_xlabel(hip)
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=6)
        ax.legend(loc="upper left", fontsize=8.5)
    ejes[0].set_ylabel("Macro F1 (validación cruzada por grupos, solo entrenamiento)")
    st.encabezado(ejes[0], "4.17", "Ajuste de hiperparámetros", subir=26)
    st.derecha(ejes[1], "estrella = valor elegido; línea roja = azar", 26)
    fig.tight_layout(w_pad=1.4)
    st.guardar(fig, dir_fig, "4_17_ajuste_hiperparametros")


# ------------------------------------------------------------ 4.18 ROC y PR
def _curvas(y, scores):
    k = scores.shape[1]
    onehot = np.eye(k)[y]
    fpr, tpr, _ = roc_curve(onehot.ravel(), scores.ravel())
    pr, rc, _ = precision_recall_curve(onehot.ravel(), scores.ravel())
    return (fpr, tpr, auc(fpr, tpr)), (rc, pr, average_precision_score(onehot.ravel(), scores.ravel()))


def fig_4_18(dir_tablas, dir_fig):
    fuentes = []
    for archivo, etiqueta_fila in [("E1_scores", "Partición oficial"), ("E3_scores_rep0", "Grupos de textos (repetición 1)")]:
        ruta = os.path.join(dir_tablas, archivo + ".npz")
        if os.path.exists(ruta):
            d = np.load(ruta)
            meta = json.load(open(os.path.join(dir_tablas, archivo + "_meta.json"), encoding="utf-8"))
            fuentes.append((etiqueta_fila, d["y"], [d[f"s{i}"] for i in range(len(meta["modelos"]))], meta["modelos"]))
    fig, ejes = plt.subplots(len(fuentes), 2, figsize=(9.0, 4.2 * len(fuentes)), squeeze=False)
    for fila, (nombre_fila, y, mats, modelos) in enumerate(fuentes):
        k = mats[0].shape[1]
        estilos = ["-", "--", "-.", ":"]                # grosor y trazo distintos: las curvas perfectas se superponen
        for i_m, (n, s) in enumerate(zip(modelos, mats)):
            (fpr, tpr, a_roc), (rc, pr, ap) = _curvas(y, s)
            kw = dict(lw=3.0 - 0.6 * i_m, ls=estilos[i_m % 4], color=color_modelo(n))
            ejes[fila, 0].plot(fpr, tpr, label=f"{etiqueta(n)} (AUC = {a_roc:.2f})", **kw)
            ejes[fila, 1].plot(rc, pr, label=f"{etiqueta(n)} (AP = {ap:.2f})", **kw)
        ejes[fila, 0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.0)
        ejes[fila, 1].axhline(1 / k, color=GRIS, ls="--", lw=1.0)
        for j, (xl, yl, tt) in enumerate([("Tasa de falsos positivos", "Tasa de verdaderos positivos", "Curva ROC"),
                                           ("Recall", "Precision", "Curva Precision-Recall")]):
            ax = ejes[fila, j]
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True)
            ubicacion = {(0, 0): "lower right", (0, 1): "lower left", (1, 0): "upper left", (1, 1): "upper right"}
            ax.legend(loc=ubicacion[(min(fila, 1), j)], fontsize=8)
            ax.set_title(f"{nombre_fila}: {tt}", fontsize=10, fontweight="bold", pad=6)
    st.encabezado(ejes[0, 0], "4.18", "Curvas ROC y Precision-Recall (micro-promedio)", subir=30)
    fig.tight_layout(h_pad=2.0, w_pad=1.6)
    st.guardar(fig, dir_fig, "4_18_curvas_roc_pr")


# ------------------------------------------------- 4.19 representaciones E8
def fig_4_19(dir_tablas, dir_fig):
    e8 = _t(dir_tablas, "E8_representaciones.csv")
    reps = list(dict.fromkeys(e8["representacion"]))
    colores = [AZUL, NARANJA, MORADO]
    fig, ejes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    for ax, esquema, titulo in [(ejes[0], "partición oficial", "(a) Partición oficial 80/20"),
                                (ejes[1], "grupos de textos", "(b) Validación por grupos de textos")]:
        g = e8[e8["esquema"] == esquema]
        ancho = 0.8 / len(reps)
        for j, rep in enumerate(reps):
            h = g[g["representacion"] == rep].set_index("modelo").reindex(OFICIALES)
            x = np.arange(len(OFICIALES)) + (j - (len(reps) - 1) / 2) * ancho
            barras = ax.bar(x, h["f1_media"], width=ancho * 0.92, color=colores[j % 3], edgecolor=GRIS, linewidth=0.6,
                            label=rep, zorder=3)
            for b, v in zip(barras, h["f1_media"]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(_azar(3), color=ROJO, ls="--", lw=1.0)
        ax.set_xticks(range(len(OFICIALES)))
        ax.set_xticklabels([etiqueta(n).replace(" ", "\n", 1) for n in OFICIALES], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y")
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=6)
    ejes[0].set_ylabel("Macro F1")
    st.encabezado(ejes[0], "4.19", "Sensibilidad a la representación del texto", subir=26)
    st.derecha(ejes[1], "línea roja = azar", 26)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.07, 1, 1))
    fig.legend(handles=[Patch(facecolor=colores[j % 3], edgecolor=GRIS, label=rep) for j, rep in enumerate(reps)],
               loc="lower center", ncol=len(reps), fontsize=9)
    st.guardar(fig, dir_fig, "4_19_sensibilidad_representacion")


FIGURAS = [("E1_metricas.csv", fig_4_8), ("E1_matrices_confusion.csv", fig_4_9), ("E2_semillas.csv", fig_4_10),
           ("E3_grupos_resumen.csv", fig_4_11), ("E3_loto_por_texto.csv", fig_4_12),
           ("E6_tiempos_resumen.csv", fig_4_13), (("E1_metricas.csv", "E3_grupos_resumen.csv", "E6_tiempos_resumen.csv"), fig_4_14),
           ("E7_escalamiento.csv", fig_4_15), ("E2_semillas.csv", fig_4_16), ("E5_ajuste.csv", fig_4_17),
           ("E1_scores.npz", fig_4_18), ("E8_representaciones.csv", fig_4_19)]


def generar_todas(dir_tablas, dir_fig):
    """Genera las figuras cuyas tablas existen; avisa de las que no pudo generar."""
    hechas = 0
    with plt.rc_context(st.ESTILO):
        for requisitos, funcion in FIGURAS:
            req = [requisitos] if isinstance(requisitos, str) else list(requisitos)
            faltan = [r for r in req if not os.path.exists(os.path.join(dir_tablas, r))]
            if faltan:
                print(f"  (omitida {funcion.__name__}: faltan {', '.join(faltan)})")
                continue
            funcion(dir_tablas, dir_fig)
            hechas += 1
    return hechas