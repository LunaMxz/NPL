"""
Gráficas de la investigación (figuras 4.1 a 4.7), con un mismo estilo para todas.

    4.1  Distribución de la variable de sentimiento             (salida de la Etapa 1)
    4.2  Comentarios por materia y clase                        (salida de la Etapa 1)
    4.3  Distribución acumulada (ECDF) de la longitud           (salida de la Etapa 2)
    4.4  Términos más frecuentes por clase                      (salida de la Etapa 2)
    4.5  Varianza explicada acumulada (PCA) de la matriz TF-IDF (salida de la Etapa 3)
    4.6  IDF frente a frecuencia documental                     (salida de la Etapa 3)
    4.7  Diagnóstico de las dos particiones (opcional)          (salidas de la Etapa 3)

Uso (desde la carpeta NPL DEPLOV):
    python analisis_graficas.py --limpio "Etapa 1/salida_pipeline/feedback_training_limpio.csv" --preprocesado "Etapa 2/salida_etapa2/feedback_training_preprocesado.csv" --etapa3 "Etapa 3/salida_estratificada" --etapa3-por-texto "Etapa 3/salida_por_texto" --salida ./figuras

--etapa3-por-texto es opcional; sin él no se genera la figura 4.7.
Requiere: pandas, numpy, matplotlib, scikit-learn.
"""
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.decomposition import PCA

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ------------------------------------------------------------ configuración
INCLUIR_TITULO = True        # título en negrita con insignia numerada dentro de la figura
FONDO_FIGURA = "#F2F4F6"     # fondo gris muy claro; usa "white" para fondo blanco puro
FORMATOS = ("png", "pdf")
TOP_TERMINOS = 10

# Paleta (azul, rojo, naranja y verde de uso clásico en ciencia de datos)
AZUL, ROJO, NARANJA = "#1F6FBF", "#D62728", "#F5A031"
MORADO, VERDE_INSIGNIA, GRIS = "#8E6BBE", "#1B5E45", "#333333"

ORDEN_CLASES = ["positive", "neutral", "negative", "irrelevant"]
NOMBRE_CLASE = {"positive": "Positivo", "neutral": "Neutral",
                "negative": "Negativo", "irrelevant": "Irrelevante"}
COLOR_CLASE = {"positive": AZUL, "neutral": NARANJA, "negative": ROJO, "irrelevant": MORADO}
COLOR_TEXTO_CLASE = {"positive": "white", "neutral": "black",
                     "negative": "white", "irrelevant": "white"}

ESTILO = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.facecolor": FONDO_FIGURA, "savefig.facecolor": "auto",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.axisbelow": True,
    "grid.color": "#D3D8DE", "grid.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.color": GRIS, "ytick.color": GRIS,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "#B8BEC6",
    "legend.fancybox": False, "legend.framealpha": 0.95,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "pdf.fonttype": 42,
}


# --------------------------------------------------------------- utilidades
def _nombre(c):
    return NOMBRE_CLASE.get(c, str(c).capitalize())


def _color(c):
    return COLOR_CLASE.get(c, "#56B4E9")


def _clases(serie):
    p = set(serie.unique())
    return [c for c in ORDEN_CLASES if c in p] + sorted(p - set(ORDEN_CLASES))


def _miles(x, _pos=None):
    return f"{int(round(x)):,}"


def _porcentaje(x, _pos=None):
    return f"{x * 100:.0f} %"


def _guardar(fig, outdir, nombre):
    for ext in FORMATOS:
        fig.savefig(os.path.join(outdir, f"{nombre}.{ext}"))
    plt.close(fig)


def _derecha(ax, texto, subir=22):
    """Texto pequeño alineado a la derecha, sobre el borde superior del eje."""
    ax.annotate(texto, xy=(1, 1), xycoords="axes fraction",
                xytext=(0, subir if INCLUIR_TITULO else 7),
                textcoords="offset points", ha="right", va="center",
                fontsize=9.5, color=GRIS, annotation_clip=False)


def _encabezado(ax, numero, titulo, derecha=None, subir=22):
    """Título en negrita con insignia numerada, sobre el borde superior del eje."""
    if INCLUIR_TITULO:
        ax.annotate(numero, xy=(0, 1), xycoords="axes fraction", xytext=(2, subir),
                    textcoords="offset points", ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color="white", annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.95",
                              fc=VERDE_INSIGNIA, ec="none"))
        ax.annotate(titulo, xy=(0, 1), xycoords="axes fraction", xytext=(34, subir),
                    textcoords="offset points", ha="left", va="center", fontsize=13,
                    fontweight="bold", color="#1A1A1A", annotation_clip=False)
    if derecha:
        _derecha(ax, derecha, subir)


# ------------------------------------- 4.1 Distribución de sentimiento
def figura_sentimiento(df, outdir):
    conteo = df["sentimiento"].value_counts()
    clases = _clases(df["sentimiento"])
    valores = [int(conteo[c]) for c in clases]
    total = sum(valores)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    x = np.arange(len(clases))
    barras = ax.bar(x, valores, width=0.55, color=[_color(c) for c in clases],
                    edgecolor=GRIS, linewidth=0.7, zorder=3)
    ax.grid(axis="y")
    for barra, v in zip(barras, valores):
        ax.annotate(f"{v:,}\n({v / total * 100:.1f} %)",
                    xy=(barra.get_x() + barra.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, linespacing=1.25)
    ax.set_xticks(x)
    ax.set_xticklabels([_nombre(c) for c in clases])
    ax.set_ylim(0, max(valores) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_xlabel("Clase de sentimiento")
    ax.set_ylabel("Número de comentarios")
    _encabezado(ax, "4.1", "Distribución de la variable de sentimiento", f"n = {total:,}")
    _guardar(fig, outdir, "4_1_distribucion_sentimiento")


# ---------------------------------------------- 4.2 Comentarios por materia
def figura_materias(df, outdir):
    clases = _clases(df["sentimiento"])
    tabla = pd.crosstab(df["materia"], df["sentimiento"]).reindex(columns=clases, fill_value=0)
    totales = tabla.sum(axis=1)
    orden = totales.sort_values(ascending=True, kind="stable").index   # mayor arriba
    tabla, totales = tabla.loc[orden], totales.loc[orden]

    n_materias = len(tabla)
    fig, ax = plt.subplots(figsize=(7.4, max(4.2, 0.36 * n_materias + 2.0)))
    y = np.arange(n_materias)
    acumulado = np.zeros(n_materias)
    maximo = float(totales.max())

    for clase in clases:
        valores = tabla[clase].to_numpy(dtype=float)
        ax.barh(y, valores, left=acumulado, height=0.72, color=_color(clase),
                edgecolor="white", linewidth=0.8, zorder=3)
        for yi, (inicio, v) in enumerate(zip(acumulado, valores)):
            if v >= 0.06 * maximo:   # solo si el número cabe dentro del segmento
                ax.text(inicio + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=8.5, color=COLOR_TEXTO_CLASE.get(clase, "black"), zorder=4)
        acumulado += valores
    for yi, t in enumerate(totales):
        ax.text(t + maximo * 0.012, yi, f"{int(t):,}", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=GRIS)

    ax.set_yticks(y)
    ax.set_yticklabels(tabla.index)
    ax.set_ylim(-0.6, n_materias - 0.4)
    ax.set_xlim(0, maximo * 1.1)
    ax.grid(axis="x")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(_miles))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Número de comentarios")
    ax.set_ylabel("Materia")
    manijas = [Patch(facecolor=_color(c), edgecolor=GRIS, linewidth=0.6, label=_nombre(c))
               for c in clases]
    ax.legend(handles=manijas, loc="lower right")
    _encabezado(ax, "4.2", "Comentarios por materia y clase", f"n = {int(totales.sum()):,}")
    _guardar(fig, outdir, "4_2_comentarios_por_materia")


# ------------------------------------------------- 4.3 ECDF de la longitud
def figura_ecdf_longitud(df, outdir):
    antes = df["comentario_limpio"].astype(str).str.split().str.len().to_numpy()
    despues = df["comentario_procesado"].astype(str).str.split().str.len().to_numpy()

    def ecdf(v):
        x = np.sort(v)
        return np.concatenate(([0], x)), np.concatenate(([0], np.arange(1, len(x) + 1) / len(x)))

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for v, color, etiqueta in [(antes, AZUL, "Antes"), (despues, ROJO, "Después")]:
        x, y = ecdf(v)
        ax.step(x, y, where="post", color=color, lw=2.0,
                label=f"{etiqueta} (media = {v.mean():.2f})")
    ax.grid(True)
    ax.set_xlim(0, max(antes.max(), despues.max()) + 0.5)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(_porcentaje))
    ax.set_xlabel("Longitud del comentario (tokens separados por espacio)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.legend(loc="upper left")
    _encabezado(ax, "4.3", "Distribución acumulada de la longitud", f"n = {len(df):,}")
    _guardar(fig, outdir, "4_3_ecdf_longitud")


# --------------------------------------- 4.4 Términos frecuentes por clase
def figura_terminos_por_clase(df, outdir):
    clases = _clases(df["sentimiento"])
    fig, ejes = plt.subplots(1, len(clases), figsize=(3.5 * len(clases), 4.4))
    ejes = np.atleast_1d(ejes)
    for ax, clase in zip(ejes, clases):
        textos = df.loc[df["sentimiento"] == clase, "comentario_procesado"].astype(str)
        conteo = {}
        for t in textos:                       # frecuencia = comentarios que contienen el término
            for termino in set(t.split()):
                conteo[termino] = conteo.get(termino, 0) + 1
        top = sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_TERMINOS]
        terminos = [t for t, _ in top][::-1]
        valores = [v for _, v in top][::-1]
        y = np.arange(len(terminos))
        ax.barh(y, valores, height=0.7, color=_color(clase), edgecolor=GRIS,
                linewidth=0.6, zorder=3)
        for yi, v in zip(y, valores):
            ax.text(v + max(valores) * 0.015, yi, f"{v:,}", va="center", ha="left", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(terminos)
        ax.set_xlim(0, max(valores) * 1.2)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(_miles))
        ax.set_title(f"{_nombre(clase)} (n = {len(textos):,})", fontsize=10.5,
                     fontweight="bold", pad=8)
    _encabezado(ejes[0], "4.4", "Términos más frecuentes por clase", subir=40)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.supxlabel("Número de comentarios que contienen el término", fontsize=10.5, y=0.01)
    _guardar(fig, outdir, "4_4_terminos_frecuentes_por_clase")


# -------------------------------------------- 4.5 Varianza explicada (PCA)
def figura_varianza_pca(X, outdir):
    pca = PCA(svd_solver="full").fit(X)
    razones = pca.explained_variance_ratio_
    acum = np.cumsum(razones)
    k90 = int(np.searchsorted(acum, 0.90) + 1)
    k95 = int(np.searchsorted(acum, 0.95) + 1)
    m = min(len(razones), max(20, k95 + 3))
    x = np.arange(1, m + 1)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(x, razones[:m], width=0.65, color=NARANJA, edgecolor=GRIS, linewidth=0.4,
           zorder=3, label="Varianza de cada componente")
    ax.plot(x, acum[:m], color=AZUL, lw=1.8, marker="o", ms=4.5, zorder=4,
            label="Varianza acumulada")
    for xi, v in zip(x[:10], acum[:10]):
        ax.annotate(f"{v * 100:.0f}%", xy=(xi, v), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=GRIS, zorder=5)
    ax.axhline(0.90, color=GRIS, ls="--", lw=0.9, zorder=2)
    ax.axvline(k90, color=AZUL, ls="--", lw=1.0, zorder=2)
    ax.annotate(f"90 % de la varianza\ncon {k90} componentes",
                xy=(k90, acum[k90 - 1]), xytext=(k90 + m * 0.14, 0.52),
                fontsize=9.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=0.9))
    ax.grid(True)
    ax.set_xlim(0.3, m + 0.7)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1] + list(range(5, m + 1, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(_porcentaje))
    ax.set_xlabel("Número de componentes principales")
    ax.set_ylabel("Varianza explicada")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.09))
    _encabezado(ax, "4.5", "Varianza explicada acumulada (PCA)",
                f"{X.shape[0]:,} documentos × {X.shape[1]:,} términos")
    _guardar(fig, outdir, "4_5_varianza_explicada_pca")
    return k90, k95


# ----------------------------------------------------- 4.6 IDF y frecuencia
def figura_idf(idf_df, n_train, outdir):
    idf_df = idf_df.copy()
    idf_df["df"] = np.rint(n_train / 10 ** idf_df["idf"]).astype(int)
    idf_df["bigrama"] = idf_df["termino"].str.contains(" ")
    uni, bi = idf_df[~idf_df["bigrama"]], idf_df[idf_df["bigrama"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    xs = np.logspace(0, np.log10(n_train), 300)
    a1.plot(xs, np.log10(n_train / xs), color=ROJO, lw=1.8, zorder=2,
            label=r"$\mathrm{IDF}(t)=\log_{10}(N/df(t))$")
    a1.scatter(uni["df"], uni["idf"], s=26, color=AZUL, alpha=0.9, zorder=3,
               edgecolor="white", linewidth=0.5, label=f"Unigramas ({len(uni):,})")
    if len(bi):
        a1.scatter(bi["df"], bi["idf"], s=30, marker="^", color=NARANJA, alpha=0.95, zorder=3,
                   edgecolor="white", linewidth=0.5, label=f"Bigramas ({len(bi):,})")
    a1.set_xscale("log")
    a1.xaxis.set_major_formatter(FuncFormatter(_miles))
    a1.set_xlim(0.8, n_train * 1.25)
    a1.set_ylim(-0.08, np.log10(n_train) + 0.15)
    a1.grid(True)
    a1.set_xlabel("Frecuencia documental $df(t)$ (escala logarítmica)")
    a1.set_ylabel("IDF$(t)$")
    a1.legend(loc="upper right")

    bordes = np.arange(0, np.log10(n_train) + 0.1, 0.1)
    series = [uni["idf"]] + ([bi["idf"]] if len(bi) else [])
    colores = [AZUL] + ([NARANJA] if len(bi) else [])
    etiquetas = ["Unigramas"] + (["Bigramas"] if len(bi) else [])
    a2.hist(series, bins=bordes, stacked=True, color=colores, label=etiquetas,
            edgecolor="white", linewidth=0.6, zorder=3)
    a2.grid(axis="y")
    a2.yaxis.set_major_locator(MaxNLocator(integer=True))
    a2.set_xlabel("IDF$(t)$")
    a2.set_ylabel("Número de términos")
    a2.legend(loc="upper left")
    _encabezado(a1, "4.6", "IDF frente a frecuencia documental")
    _derecha(a2, f"N = {n_train:,} documentos")
    fig.tight_layout(w_pad=2.5)
    _guardar(fig, outdir, "4_6_idf_vs_frecuencia_documental")


# ------------------------------------ 4.7 Diagnóstico de las particiones
def figura_diagnostico_particiones(carpeta_estratificada, carpeta_por_texto, outdir):
    """Compara las dos particiones: fuga de textos y vectores de prueba sin información."""
    def resumen(carpeta):
        with open(os.path.join(carpeta, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        prueba = pd.read_csv(os.path.join(carpeta, "tfidf_test.csv"))
        nulos = int((prueba.abs().sum(axis=1) == 0).sum())
        return {"compartidos": c["textos_compartidos_train_test"],
                "textos_test": c["textos_distintos_test"],
                "nulos": nulos, "n_test": len(prueba)}

    a, b = resumen(carpeta_estratificada), resumen(carpeta_por_texto)
    etiquetas = ["Estratificada\n(80/20, propuesta)", "Por texto\n(diagnóstico)"]
    colores = [AZUL, NARANJA]
    paneles = [
        ("(a) Textos de prueba que también\nestán en entrenamiento",
         [a["compartidos"] / a["textos_test"] * 100, b["compartidos"] / b["textos_test"] * 100],
         [f"{a['compartidos']} de {a['textos_test']}", f"{b['compartidos']} de {b['textos_test']}"]),
        ("(b) Comentarios de prueba con\nvector TF-IDF nulo",
         [a["nulos"] / a["n_test"] * 100, b["nulos"] / b["n_test"] * 100],
         [f"{a['nulos']} de {a['n_test']}", f"{b['nulos']} de {b['n_test']}"]),
    ]
    fig, ejes = plt.subplots(1, 2, figsize=(7.8, 4.5))
    for ax, (titulo, valores, rotulos) in zip(ejes, paneles):
        barras = ax.bar(range(2), valores, width=0.55, color=colores,
                        edgecolor=GRIS, linewidth=0.7, zorder=3)
        for barra, v, r in zip(barras, valores, rotulos):
            ax.annotate(f"{v:.1f} %\n({r})", xy=(barra.get_x() + barra.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9.5, linespacing=1.25)
        ax.set_xticks(range(2))
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_ylim(0, 125)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f} %"))
        ax.grid(axis="y")
        ax.set_title(titulo, fontsize=10, pad=8)
    ejes[0].set_ylabel("Porcentaje")
    _encabezado(ejes[0], "4.7", "Diagnóstico de las particiones", subir=40)
    fig.tight_layout(w_pad=2.0)
    _guardar(fig, outdir, "4_7_diagnostico_particiones")


# ----------------------------------------------------------- punto de entrada
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limpio", required=True,
                    help="CSV de la Etapa 1 (feedback_training_limpio.csv)")
    ap.add_argument("--preprocesado", required=True,
                    help="CSV de la Etapa 2 (feedback_training_preprocesado.csv)")
    ap.add_argument("--etapa3", required=True,
                    help="Carpeta de la partición oficial de la Etapa 3 (salida_estratificada)")
    ap.add_argument("--etapa3-por-texto", default=None,
                    help="Carpeta de la partición por texto (salida_por_texto); habilita la figura 4.7")
    ap.add_argument("--salida", default="./figuras")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    limpio = pd.read_csv(args.limpio)
    for col in ["materia", "sentimiento"]:
        if col not in limpio.columns:
            raise KeyError(f"Falta la columna '{col}' en {args.limpio}")
    df = pd.read_csv(args.preprocesado)
    for col in ["comentario_limpio", "comentario_procesado", "sentimiento"]:
        if col not in df.columns:
            raise KeyError(f"Falta la columna '{col}' en {args.preprocesado}")
    X = pd.read_csv(os.path.join(args.etapa3, "tfidf_train.csv")).to_numpy(dtype=float)
    idf_df = pd.read_csv(os.path.join(args.etapa3, "idf.csv"))
    with open(os.path.join(args.etapa3, "config.json"), encoding="utf-8") as f:
        n_train = int(json.load(f)["n_train"])

    print("=== DIAGNOSTICO DEL CORPUS ===")
    print(f"Comentarios: {len(df)}")
    print(f"Textos distintos (comentario_limpio): {df['comentario_limpio'].nunique()}")
    print(f"Textos distintos (comentario_procesado): {df['comentario_procesado'].nunique()}")

    with plt.rc_context(ESTILO):
        figura_sentimiento(limpio, args.salida)
        figura_materias(limpio, args.salida)
        figura_ecdf_longitud(df, args.salida)
        figura_terminos_por_clase(df, args.salida)
        k90, k95 = figura_varianza_pca(X, args.salida)
        figura_idf(idf_df, n_train, args.salida)
        n_fig = 6
        if args.etapa3_por_texto:
            figura_diagnostico_particiones(args.etapa3, args.etapa3_por_texto, args.salida)
            n_fig = 7

    print(f"PCA: {k90} componentes explican 90 % y {k95} explican 95 % de la varianza")
    print(f"{n_fig} figuras guardadas en: {args.salida}  ({', '.join(FORMATOS)})")


if __name__ == "__main__":
    main()