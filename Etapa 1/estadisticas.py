"""
Estadisticas descriptivas del corpus limpio (Etapa 1). Solo estadisticas:
las graficas viven en analisis_graficas.

Se puede usar de dos formas:

  1) Dentro del pipeline (pipeline.py la llama igual que antes).
  2) Directo, sobre una salida ya generada (sin volver a correr el pipeline):
         python estadisticas.py --input salida_pipeline/feedback_training_limpio.csv --outdir salida_pipeline

Archivos que genera en --outdir:
  estadisticas_etapa1.json, estadisticas_por_materia.csv, textos_repetidos.csv
"""
import argparse
import json
import os
import re

import pandas as pd

import config


def contar_tokens(texto):
    return len(re.findall(r"\b\w+\b", str(texto).lower()))


def _estadisticas_longitud(df):
    n = df["n_tokens"]
    return {
        "promedio": round(float(n.mean()), 2),
        "mediana": float(n.median()),
        "desviacion_estandar": round(float(n.std()), 2),
        "minimo": int(n.min()),
        "maximo": int(n.max()),
    }


def _diversidad_textual(df):
    """Cuantos textos distintos hay y si algun texto aparece con etiquetas contradictorias."""
    total = len(df)
    por_texto = df.groupby("comentario_limpio")["sentimiento"].agg(
        veces="size", clases="nunique", clase=lambda s: s.mode().iat[0])
    distintos = int(len(por_texto))
    return por_texto, {
        "comentarios": int(total),
        "textos_distintos": distintos,
        "porcentaje_textos_distintos": round(distintos / total * 100, 2),
        "comentarios_con_texto_repetido": int((df.duplicated("comentario_limpio", keep=False)).sum()),
        "textos_con_etiquetas_contradictorias": int((por_texto["clases"] > 1).sum()),
        "veces_texto_mas_repetido": int(por_texto["veces"].max()),
        "textos_distintos_por_clase": {
            k: int(v) for k, v in
            df.groupby("sentimiento")["comentario_limpio"].nunique().items()},
    }


def generar_estadisticas(df, outdir, etiqueta_muestra=""):
    os.makedirs(outdir, exist_ok=True)
    df["n_tokens"] = df["comentario_limpio"].map(contar_tokens)

    total = len(df)
    distribucion = df["sentimiento"].value_counts()
    por_materia = pd.crosstab(df["materia"], df["sentimiento"])
    por_materia["total"] = por_materia.sum(axis=1)
    por_materia = por_materia.sort_values("total", ascending=False)
    por_texto, diversidad = _diversidad_textual(df)
    longitud = _estadisticas_longitud(df)

    print(f"Estadisticas generales del corpus tras el preprocesamiento{etiqueta_muestra}:")
    print(f"- Total de comentarios: {total}")
    print(f"- Longitud promedio por comentario (tokens): {longitud['promedio']:.2f}")
    print("- Distribucion de clases:", ", ".join(f"{k} = {v}" for k, v in distribucion.items()))
    print(f"- Materias: {df['materia'].nunique()}")
    print(f"- Textos distintos: {diversidad['textos_distintos']} de {total} "
          f"({diversidad['porcentaje_textos_distintos']:.2f} %)")
    print(f"- Textos con etiquetas contradictorias: {diversidad['textos_con_etiquetas_contradictorias']}")
    print(f"- Veces que se repite el texto mas frecuente: {diversidad['veces_texto_mas_repetido']}")

    resumen = {
        "total_comentarios": int(total),
        "longitud_tokens": longitud,
        "distribucion_clases": {
            k: {"n": int(v), "porcentaje": round(v / total * 100, 2)}
            for k, v in distribucion.items()},
        "n_materias": int(df["materia"].nunique()),
        "diversidad_textual": diversidad,
    }
    with open(os.path.join(outdir, config.ARCHIVO_ESTADISTICAS), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    por_materia.to_csv(os.path.join(outdir, config.ARCHIVO_POR_MATERIA))
    por_texto.rename(columns={"clase": "clase_mas_frecuente"}).sort_values(
        "veces", ascending=False).head(config.TOP_TEXTOS_REPETIDOS).to_csv(
        os.path.join(outdir, config.ARCHIVO_TEXTOS_REPETIDOS))
    return resumen


# Nombre anterior, para que pipeline.py siga funcionando sin cambios.
generar_estadisticas_y_graficas = generar_estadisticas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("salida_pipeline", config.ARCHIVO_LIMPIO))
    ap.add_argument("--outdir", default="salida_pipeline")
    args = ap.parse_args()
    generar_estadisticas(pd.read_csv(args.input), args.outdir)