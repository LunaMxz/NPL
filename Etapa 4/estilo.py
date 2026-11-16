"""Estilo común de las figuras de la Etapa 4 (el mismo de analisis_graficas.py)."""
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

INCLUIR_TITULO = True        # título en negrita con insignia numerada dentro de la figura
FONDO_FIGURA = "#F2F4F6"     # usa "white" para fondo blanco puro
FORMATOS = ("png", "pdf")

AZUL, ROJO, NARANJA, MORADO = "#1F6FBF", "#D62728", "#F5A031", "#8E6BBE"
VERDE_INSIGNIA, GRIS, GRIS_CLARO = "#1B5E45", "#333333", "#B9BEC6"

NOMBRE_CLASE = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo", "irrelevant": "Irrelevante"}
COLOR_MODELO = {"Naive Bayes": AZUL, "SVM lineal (liblinear)": ROJO,
                "SVM lineal (libsvm)": NARANJA, "Regresión Logística": MORADO}
ETIQUETA = {"Naive Bayes": "Naive Bayes", "SVM lineal (liblinear)": "SVM liblinear",
            "SVM lineal (libsvm)": "SVM libsvm", "Regresión Logística": "Reg. Logística",
            "SVM lineal (liblinear), sin normalizar": "SVM liblinear (crudo)",
            "SVM lineal (libsvm), sin normalizar": "SVM libsvm (crudo)",
            "Regresión Logística, con L2": "Reg. Logística (L2)"}

ESTILO = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.facecolor": FONDO_FIGURA, "savefig.facecolor": "auto", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.axisbelow": True,
    "grid.color": "#D3D8DE", "grid.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.color": GRIS, "ytick.color": GRIS,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": GRIS_CLARO,
    "legend.fancybox": False, "legend.framealpha": 0.95,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "pdf.fonttype": 42,
}


def etiqueta(nombre):
    return ETIQUETA.get(nombre, nombre)


def color_modelo(nombre):
    """Color por familia de modelo; las variantes de sensibilidad usan el color de su familia."""
    if nombre in COLOR_MODELO:
        return COLOR_MODELO[nombre]
    if "liblinear" in nombre:
        return ROJO
    if "libsvm" in nombre:
        return NARANJA
    if "Logística" in nombre:
        return MORADO
    return GRIS_CLARO


def guardar(fig, dir_figuras, nombre):
    os.makedirs(dir_figuras, exist_ok=True)
    for ext in FORMATOS:
        fig.savefig(os.path.join(dir_figuras, f"{nombre}.{ext}"))
    plt.close(fig)


def derecha(ax, texto, subir=22):
    ax.annotate(texto, xy=(1, 1), xycoords="axes fraction", xytext=(0, subir if INCLUIR_TITULO else 7),
                textcoords="offset points", ha="right", va="center", fontsize=9.5, color=GRIS,
                annotation_clip=False)


def encabezado(ax, numero, titulo, texto_derecha=None, subir=22):
    """Título en negrita con insignia numerada, sobre el borde superior del eje."""
    if INCLUIR_TITULO:
        ax.annotate(numero, xy=(0, 1), xycoords="axes fraction", xytext=(2, subir),
                    textcoords="offset points", ha="left", va="center", fontsize=9.5, fontweight="bold",
                    color="white", annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.95", fc=VERDE_INSIGNIA, ec="none"))
        ax.annotate(titulo, xy=(0, 1), xycoords="axes fraction", xytext=(34, subir),
                    textcoords="offset points", ha="left", va="center", fontsize=13, fontweight="bold",
                    color="#1A1A1A", annotation_clip=False)
    if texto_derecha:
        derecha(ax, texto_derecha, subir)