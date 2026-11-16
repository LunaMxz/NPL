"""
ETAPA 5 - Paso 3: figuras 5.1 a 5.4 a partir de las tablas de experimentos_general.py.

    python graficas_general.py --salida ./salida_etapa5

5.1  Macro F1 en el corpus general (agrupado por id), con IC 95 %
5.2  Matrices de confusión en el corpus general
5.3  Transferencia a comentarios de estudiantes (Macro F1 y exactitud por texto)
5.4  Costo computacional: tiempo de entrenamiento frente a n, y desempeño frente a tiempo
"""
import argparse
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

AZUL, ROJO, NARANJA, MORADO, GRIS, GRIS_CLARO = "#1F6FBF", "#D62728", "#F5A031", "#8E6BBE", "#333333", "#B9BEC6"
COLOR = {"Naive Bayes": AZUL, "SVM lineal (liblinear)": ROJO, "SVM lineal (libsvm)": NARANJA,
         "Regresión Logística": MORADO}
ETIQ = {"Naive Bayes": "Naive Bayes", "SVM lineal (liblinear)": "SVM liblinear", "SVM lineal (libsvm)": "SVM libsvm",
        "Regresión Logística": "Reg. Logística"}
MODELOS = list(COLOR)
REFS = ["Azar", "Clase mayoritaria"]
ESTILO = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
          "font.size": 10, "axes.facecolor": "white", "figure.facecolor": "#F2F4F6", "savefig.facecolor": "auto",
          "axes.edgecolor": "#444444", "grid.color": "#D3D8DE", "grid.linewidth": 0.7, "axes.axisbelow": True,
          "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42}


def _guardar(fig, carpeta, nombre):
    os.makedirs(carpeta, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(carpeta, f"{nombre}.{ext}"))
    plt.close(fig)


def _titulo(ax, numero, texto):
    ax.set_title(f"{numero}  {texto}", loc="left", fontsize=12.5, fontweight="bold", pad=10)


def _barras_ic(ax, r, col, titulo=None, azar=None):
    orden = [n for n in MODELOS + REFS if n in set(r["modelo"])]
    r = r.set_index("modelo").loc[orden]
    y = np.arange(len(orden))[::-1]
    for yi, n in zip(y, orden):
        v, lo, hi = r.loc[n, col], r.loc[n, f"{col.replace('_media', '')}_ic_inf"], r.loc[n, f"{col.replace('_media', '')}_ic_sup"]
        ax.barh(yi, v, height=0.6, color=COLOR.get(n, GRIS_CLARO), edgecolor=GRIS, linewidth=0.6, zorder=3,
                hatch=None if n in COLOR else "...")
        ax.errorbar(v, yi, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], color=GRIS, capsize=3, lw=1.1, zorder=4)
        ax.text(min(hi, 1) + 0.015, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQ.get(n, n) for n in orden])
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x")
    if azar:
        ax.axvline(azar, color=ROJO, ls="--", lw=1.1, zorder=2)
    if titulo:
        ax.set_title(titulo, fontsize=10.5, fontweight="bold")


def fig_5_1(t, f):
    r = pd.read_csv(os.path.join(t, "G1_resumen.csv"))
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    _barras_ic(ejes[0], r, "f1_media", "(a) Macro F1", azar=0.25)
    _barras_ic(ejes[1], r, "accuracy_media", "(b) Accuracy", azar=0.25)
    for ax in ejes:
        ax.set_xlabel("Media de semillas (barras de error: IC 95 % de la semilla 42)")
    _titulo(ejes[0], "5.1", "Corpus general: desempeño con partición agrupada por id")
    fig.tight_layout()
    _guardar(fig, f, "5_1_corpus_general_desempeno")


def fig_5_2(t, f):
    cm = pd.read_csv(os.path.join(t, "G1_matrices_confusion.csv"))
    clases = list(dict.fromkeys(cm["clase_real"]))
    nombres = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo", "irrelevant": "Irrelevante"}
    fig, ejes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    for ax, n in zip(ejes.ravel(), MODELOS):
        g = cm[cm["modelo"] == n]
        m = np.array([[g[(g.clase_real == a) & (g.clase_predicha == b)]["n"].iloc[0] for b in clases] for a in clases], float)
        mp = m / m.sum(axis=1, keepdims=True)
        ax.imshow(mp, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, f"{mp[i, j]:.2f}\n({int(m[i, j])})", ha="center", va="center", fontsize=8.5,
                        color="white" if mp[i, j] > 0.55 else "black")
        et = [nombres.get(c, c) for c in clases]
        ax.set_xticks(range(len(clases)))
        ax.set_xticklabels(et, fontsize=8.5, rotation=20)
        ax.set_yticks(range(len(clases)))
        ax.set_yticklabels(et, fontsize=8.5)
        ax.set_title(ETIQ[n], fontsize=10.5, fontweight="bold")
        ax.grid(False)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Clase real")
    for ax in ejes[1, :]:
        ax.set_xlabel("Clase predicha")
    fig.suptitle("5.2  Matrices de confusión normalizadas por fila (corpus general, semilla 42)",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _guardar(fig, f, "5_2_matrices_confusion_general")


def fig_5_3(t, f):
    r = pd.read_csv(os.path.join(t, "G2_transferencia.csv"))
    pt = pd.read_csv(os.path.join(t, "G2_por_texto.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    _barras_ic(a, r, "f1", "(a) Macro F1 sobre comentarios de estudiantes", azar=1 / 3)
    a.set_xlabel("IC 95 % por bootstrap de textos")
    orden = {"positive": 0, "neutral": 1, "negative": 2}
    pt = pt.assign(_o=pt["clase"].map(orden)).sort_values(["_o", "texto"]).reset_index(drop=True)
    m = pt[MODELOS].to_numpy(float)
    b.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            b.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if m[i, j] > 0.6 else "black")
    b.set_xticks(range(len(MODELOS)))
    b.set_xticklabels([ETIQ[n].replace(" ", "\n", 1) for n in MODELOS], fontsize=8.5)
    b.xaxis.tick_top()
    b.set_yticks(range(len(pt)))
    b.set_yticklabels([f"[{c[:3]}] {x[:34]}" for c, x in zip(pt["clase"], pt["texto"])], fontsize=8)
    for lab, c in zip(b.get_yticklabels(), pt["clase"]):
        lab.set_color({"positive": AZUL, "neutral": "#B9770E", "negative": ROJO}[c])
    b.set_title("(b) Exactitud por texto distinto", fontsize=10.5, fontweight="bold", pad=26)
    b.grid(False)
    fig.suptitle("5.3  Transferencia: entrenar con el corpus general, probar con comentarios de estudiantes",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _guardar(fig, f, "5_3_transferencia_estudiantes")


def fig_5_4(t, f):
    g = pd.read_csv(os.path.join(t, "G3_escalamiento.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for n in MODELOS:
        h = g[g["modelo"] == n].sort_values("n_entrenamiento_efectivo")
        al = h["alpha_escalamiento"].iloc[0]
        a.errorbar(h["n_entrenamiento_efectivo"], h["fit_s_mediana"],
                   yerr=[h["fit_s_mediana"] - h["fit_s_min"], h["fit_s_max"] - h["fit_s_mediana"]],
                   marker="o", ms=5, lw=1.8, color=COLOR[n], capsize=2, label=f"{ETIQ[n]} (α = {al:.2f})")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks([2000, 5000, 10000, 20000, 45000])
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.grid(True)
    a.set_xlabel("Documentos de entrenamiento (n)")
    a.set_ylabel("Tiempo de entrenamiento (s, reloj)")
    a.legend(fontsize=8.5, title=r"$t \propto n^{\alpha}$", title_fontsize=8.5, loc="upper left")
    a.set_title("(a) Escalamiento (mediana y rango de repeticiones)", fontsize=10.5, fontweight="bold")
    for n in MODELOS:
        h = g[g["modelo"] == n].sort_values("n_entrenamiento_efectivo")
        b.plot(h["fit_s_mediana"], h["f1"], marker="o", ms=5, lw=1.4, color=COLOR[n], label=ETIQ[n])
        u = h.iloc[-1]
        b.annotate(ETIQ[n], (u["fit_s_mediana"], u["f1"]), xytext=(6, 4), textcoords="offset points", fontsize=8.5,
                   color=COLOR[n], fontweight="bold")
    b.set_xscale("log")
    b.grid(True)
    b.set_xlabel("Tiempo de entrenamiento (s, reloj, escala log)")
    b.set_ylabel("Macro F1 (prueba, partición agrupada)")
    b.set_title("(b) Desempeño frente a costo", fontsize=10.5, fontweight="bold")
    fig.suptitle("5.4  Costo computacional (libsvm entrenado con a lo sumo "
                 f"{int(g['n_entrenamiento_efectivo'][g['modelo'] == 'SVM lineal (libsvm)'].max())} documentos)",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _guardar(fig, f, "5_4_costo_computacional")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="./salida_etapa5")
    args = ap.parse_args()
    t, f = os.path.join(args.salida, "tablas"), os.path.join(args.salida, "figuras")
    hechas = 0
    with plt.rc_context(ESTILO):
        for archivo, funcion in [("G1_resumen.csv", fig_5_1), ("G1_matrices_confusion.csv", fig_5_2),
                                 ("G2_por_texto.csv", fig_5_3), ("G3_escalamiento.csv", fig_5_4)]:
            if not os.path.exists(os.path.join(t, archivo)):
                print(f"  (omitida {funcion.__name__}: falta {archivo})")
                continue
            funcion(t, f)
            hechas += 1
    print(f"{hechas} figuras generadas en: {f}")


if __name__ == "__main__":
    main()